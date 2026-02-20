#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Type.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <filesystem>
#include <unordered_set>
#include <regex>

using namespace std;

namespace {
    llvm::cl::OptionCategory mutatorArgs("Mutator options");

    llvm::cl::opt <string> inputFile(llvm::cl::Positional,
                                     llvm::cl::desc("<inputFile>"),
                                     llvm::cl::Required,
                                     llvm::cl::value_desc("filename"),
                                     llvm::cl::cat(mutatorArgs));

    llvm::cl::opt <string> outputFolder(llvm::cl::Positional,
                                        llvm::cl::desc("<outputFileFolder>"),
                                        llvm::cl::value_desc("folder"),
                                        llvm::cl::cat(mutatorArgs));


    llvm::cl::opt <int> depth("depth",
                                        llvm::cl::desc("slice depth"),
                                        llvm::cl::value_desc("folder"),
                                        llvm::cl::cat(mutatorArgs),
                                        llvm::cl::init(-1));


    llvm::cl::list<int> patternSize(
            "pattern-size",
            llvm::cl::desc("Pattern sizes"),
            llvm::cl::value_desc("s1,s2,s3,..."),
            llvm::cl::CommaSeparated,
            llvm::cl::cat(mutatorArgs),
            llvm::cl::ZeroOrMore
    );


}

std::string getOutputSrcFilename(int ith);
bool isValidOutputPath();

static llvm::ExitOnError ExitOnErr;
std::unique_ptr<llvm::Module> openInputFile(llvm::LLVMContext &Context,
                                            const string &InputFilename) {
    auto MB =
            ExitOnErr(errorOrToExpected(llvm::MemoryBuffer::getFile(InputFilename)));
    llvm::SMDiagnostic Diag;
    auto M = getLazyIRModule(std::move(MB), Diag, Context,
            /*ShouldLazyLoadMetadata=*/true);
    if (!M) {
        Diag.print("", llvm::errs(), false);
        return 0;
    }
    ExitOnErr(M->materializeAll());
    return M;
}

std::unordered_map<unsigned, unsigned> unaryOpsCnt, binaryOpsCnt, intrinsicCnt;
unsigned saveFuncs;

std::string generateFunctionName(){
    static size_t i=0;
    auto result = std::string("auto_gen_func")+to_string(i);
    i+=1;
    return result;
}

std::string getOutputPath(std::string&& fileName){
    if(outputFolder.back() == '/'){
        return outputFolder+fileName;
    }
    return outputFolder+'/'+fileName;
}

llvm::Function* moveToFunction(llvm::LLVMContext& ctx, llvm::SmallVector<llvm::Instruction*> insts){
    std::unordered_map<llvm::Value*,size_t> valSet;
    std::unordered_map<llvm::Value*, llvm::Value*> valMapping;
    //cloning all operations into a basic block
    //and collecting function arguments
    auto bb=llvm::BasicBlock::Create(ctx);
    for(auto& inst:insts){
        auto newInst = inst->clone();
        newInst->setMetadata(llvm::LLVMContext::MD_tbaa, nullptr);
        for(size_t i=0;i<newInst->getNumOperands();++i){
            auto ithOperand = newInst->getOperand(i);
            if(valMapping.find(ithOperand)!=valMapping.end()){
                newInst->setOperand(i, valMapping[ithOperand]);
            }else{
                valSet.emplace(ithOperand, 0);
            }
        }
        newInst->insertInto(bb,bb->end());
        valMapping.emplace(inst,newInst);
    }

    llvm::SmallVector<llvm::Type*> args;
    for(auto& arg: valSet){
        arg.second = args.size();
        args.push_back(arg.first->getType());
    }
    auto functionType = llvm::FunctionType::get(insts.back()->getType(), args, false);
    auto function = llvm::Function::Create(functionType, llvm::Function::ExternalLinkage, generateFunctionName());
    bb->insertInto(function);
    for(auto it = bb->begin();it!=bb->end();++it){
        for(size_t i=0;i<it->getNumOperands();++i){
            auto ithOperand = it->getOperand(i);
            if(valSet.find(ithOperand)!=valSet.end()){
                it->setOperand(i, function->getArg(valSet[ithOperand]));
            }
        }
    }
    auto returnInst = llvm::ReturnInst::Create(ctx, &bb->back(), bb);
    return function;
}

void saveFunctionToFile(llvm::Function* func, const std::string &Path) {
    /*std::error_code EC;
    llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_Text);

    if (EC) {
        llvm::errs() << "Error opening file " << Path << ": " << EC.message() << "\n";
        return;
    }

    // Print only the function body (with definition).
    func->print(OS, nullptr);*/
    using namespace llvm;
    if (!func) {
        errs() << "Error: null Function pointer.\n";
        return;
    }

    // 1. Dump the function IR to a string
    std::string funcStr;
    {
        raw_string_ostream rso(funcStr);
        func->print(rso);
    }

    // 2. Remove all integer type patterns (i1, i8, i16, i32, i64, etc.)
    std::regex intTypeRegex(R"(i[0-9]+)");
    std::string cleaned = std::regex_replace(funcStr, intTypeRegex, "%int");

    // 3. Save to file
    std::ofstream outFile(Path);
    if (!outFile) {
        llvm::errs() << "Error: cannot open file for writing: " << Path << "\n";
        return;
    }

    outFile << cleaned;
    outFile.close();

}

void updateCntMap(std::unordered_map<unsigned, unsigned>& umap, unsigned opCode){
    if(umap.find(opCode)==umap.end()){
        umap.emplace(opCode, 0);
    }
    umap[opCode]+=1;
}

bool isCallToNonIntrinsic(const llvm::Instruction *I) {
    // Check if it's a call (could also be InvokeInst in some cases)
    if (const auto *CI = llvm::dyn_cast<llvm::CallBase>(I)) {
        if (const llvm::Function *Callee = CI->getCalledFunction()) {
            return !Callee->isIntrinsic();
        }
    }
    return false;
}

bool specialCheck(const llvm::Instruction *I) {
    return llvm::isa<llvm::PHINode>(I) || llvm::isa<llvm::LandingPadInst>(I)
           //|| isCallToNonIntrinsic(I)
           || llvm::isa<llvm::AtomicRMWInst>(I)
           ||llvm::isa<llvm::CallBase>(I)
           || llvm::isa<llvm::LoadInst>(I)
           || llvm::isa<llvm::AllocaInst>(I) || llvm::isa<llvm::StoreInst>(I)
           || !llvm::isa<llvm::Operator>(I)
           || llvm::isa<llvm::UnaryInstruction>(I)
           //|| llvm::isa<llvm::ICmpInst>(I)
           || llvm::isa<llvm::ZExtInst>(I)
           || llvm::isa<llvm::SExtInst>(I) || llvm::isa<llvm::TruncInst>(I)
           || llvm::isa<llvm::GetElementPtrInst>(I) || llvm::isa<llvm::SelectInst>(I)
           || llvm::isa<llvm::PtrToIntInst>(I);
}

void canonicalizeFunction(llvm::Function* func){
    size_t idx =0;
    for(auto arg_it = func->arg_begin();arg_it!=func->arg_end();++arg_it){
        arg_it->setName(to_string(idx));
        ++idx;
    }
    for(auto it=llvm::inst_begin(func);it!=llvm::inst_end(func);++it){
        if(it->isCommutative()){
            if(it->getOperand(0)->getName()>it->getOperand(1)->getName()){
                auto val=it->getOperand(1);
                it->setOperand(1, it->getOperand(0));
                it->setOperand(0, val);
            }
        }
        if(it->getType()!=llvm::Type::getVoidTy(func->getContext())){
            it->setName(to_string(idx));
            idx+=1;
        }
    }
}


int getDAGSize(llvm::Value* val, std::unordered_map<llvm::Value*, int>& sizeMap){
    auto it=sizeMap.find(val);
    if(it==sizeMap.end()){
        auto inst = llvm::dyn_cast_or_null<llvm::Instruction>(val);
        if(!inst){
            sizeMap.emplace(val, 0);
        }else{
            sizeMap.emplace(val, getDAGSize(inst->getOperand(0), sizeMap)+
                                 getDAGSize(inst->getOperand(1), sizeMap)+1);
        }
        return sizeMap[val];
    }else{
        return it->second;
    }
}



std::vector<llvm::Value*> enumeratePatternHelper(llvm::Value* v, int size, std::unordered_map<llvm::Value*, int>& sizeMap){
    if(!llvm::isa<llvm::Instruction>(v) || size == 0){
        return {v};
    }
    llvm::Instruction* inst = llvm::dyn_cast<llvm::Instruction>(v);
    std::vector<llvm::Value*> result;
    int lhsSize= getDAGSize(inst->getOperand(0), sizeMap), rhsSize= getDAGSize(inst->getOperand(1), sizeMap);
    int  lhsUpperBound = min(lhsSize, size-1), lhsLowerBound=max(0, size-1-rhsSize);

    for(int lhs=lhsLowerBound,rhs=size-1-lhs;lhs<=lhsUpperBound&&rhs>=0;++lhs, --rhs){
        auto lhsValues = enumeratePatternHelper(inst->getOperand(0), lhs, sizeMap);
        auto rhsValues = enumeratePatternHelper(inst->getOperand(1), rhs, sizeMap);
        for(const auto& lhsVal :lhsValues){
            for(const auto& rhsVal :rhsValues){
                auto cloneInst = inst->clone();
                cloneInst->setOperand(0, lhsVal);
                cloneInst->setOperand(1, rhsVal);
                result.push_back(cloneInst);
            }
        }
    }
    return result;
}

llvm::Function* copyToFunction(llvm::Instruction* source){
    auto& context = source->getContext();
    llvm::SmallVector<llvm::Instruction*> insts, stack{source};
    while(!stack.empty()){
        auto inst = stack.back();
        stack.pop_back();
        insts.push_back(inst);
        for(int i=0;i<2;++i){

            if(auto operand = llvm::dyn_cast_or_null<llvm::Instruction>(inst->getOperand(i));operand&&operand->getParent()==nullptr){
                stack.push_back(llvm::dyn_cast<llvm::Instruction>(operand));
            }
        }
    }
    reverse(insts.begin(), insts.end());
    return moveToFunction(context, insts);
}

llvm::Value* getUniqueReturn(llvm::Function* func){
    llvm::Value *uniqueResult = nullptr;
    for (llvm::BasicBlock &BB : *func) {
        if (llvm::ReturnInst *RI = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
            llvm::Value *RV = RI->getReturnValue();
            if (!RV) assert(false && "Return void type");
            if (!uniqueResult) uniqueResult = RV;
            else if (uniqueResult != RV) assert(false && "Return multiple value");
        }
    }
    if (!uniqueResult) assert(false && "Return void type");
    return uniqueResult;
}


std::vector<llvm::Function*> enumeratePatternWithSize(llvm::Function* func, int size, std::unordered_map<llvm::Value*, int>& sizeMap){
    llvm::Value *uniqueResult = getUniqueReturn(func);
    auto result= enumeratePatternHelper(uniqueResult, size, sizeMap);
    std::vector<llvm::Function*> funcResult;
    for(auto val:result){

        if(auto inst = llvm::dyn_cast_or_null<llvm::Instruction>(val);inst){
            funcResult.push_back(copyToFunction(inst));
        }
    }
    return funcResult;
}

void sliceInstruction(llvm::Instruction* inst, llvm::SmallVector<llvm::Instruction*>& insts, int depth){
    if(depth!=0&&inst->getType()->isIntegerTy()&&!specialCheck(inst)){
        insts.push_back(inst);
        for(size_t i=0;i<inst->getNumOperands();++i){
            auto operand_inst = llvm::dyn_cast<llvm::Instruction>(inst->getOperand(i));
            if(operand_inst){
                sliceInstruction(operand_inst, insts, depth-1);
            }
        }
    }
}

void walkModule(std::shared_ptr<llvm::Module> module, int depth, const std::vector<int>& patternSizeVec){
    bool shouldSlice = depth >1;
    llvm::SmallVector<llvm::Instruction*> insts;
    for(llvm::Function& func:*module){
        for(auto it=llvm::inst_begin(func), end_it = llvm::inst_end(func);it!=end_it;++it){
            //Slice on the current instruction
            if(shouldSlice && !it->isTerminator()&& !specialCheck(&*it)){
                sliceInstruction(&*it, insts, depth);
                if(insts.size()>=1){
                    std::reverse(insts.begin(), insts.end());
                    auto func = moveToFunction(module->getContext(), insts);
                    canonicalizeFunction(func);
                    auto outputPath = getOutputPath(func->getName().str());
                    func->setName("tmp");
                    //exclude the last return instruction
                    auto funcSize = func->getInstructionCount()-1;
                    std::unordered_map<llvm::Value*, int> sizeMap;
                    if(!patternSizeVec.empty()){
                        //llvm::errs()<<"AAAAAA"<<funcSize<<"\n";
                        for(int i=0;i<patternSizeVec.size()&&patternSizeVec[i]<=funcSize;++i){
                            //llvm::errs()<<"Current size: "<<patternSizeVec[i]<<"\n";

                            auto patterns = enumeratePatternWithSize(func, patternSizeVec[i], sizeMap);
                            for(auto pattern:patterns){

                                canonicalizeFunction(pattern);
                                outputPath = getOutputPath(pattern->getName().str());
                                pattern->setName("tmp");
                                saveFunctionToFile(pattern, outputPath);
                                ++saveFuncs;
                            }
                        }
                    }else{
                        saveFunctionToFile(func,outputPath);
                        ++saveFuncs;
                    }
                }
                insts.clear();
            }

            //count ops
            unsigned opCode = it->getOpcode();
            if(auto callInst = llvm::dyn_cast<llvm::CallBase>(&*it);callInst){
                auto intrinsicID = callInst->getIntrinsicID();
                if(intrinsicID != llvm::Intrinsic::not_intrinsic){
                    updateCntMap(intrinsicCnt, intrinsicID);
                }
            }else if(it->isUnaryOp()){
                updateCntMap(unaryOpsCnt, opCode);
            }else if(it->isBinaryOp()){
                updateCntMap(binaryOpsCnt, opCode);
            }
        }
    }
}

std::string getUnaryOrBinaryOpName(unsigned opcode) {

    switch (opcode) {
        // Unary
        case llvm::Instruction::FNeg:   return "fneg";

            // Binary (integer)
        case llvm::Instruction::Add:    return "add";
        case llvm::Instruction::Sub:    return "sub";
        case llvm::Instruction::Mul:    return "mul";
        case llvm::Instruction::UDiv:   return "udiv";
        case llvm::Instruction::SDiv:   return "sdiv";
        case llvm::Instruction::URem:   return "urem";
        case llvm::Instruction::SRem:   return "srem";

            // Binary (floating point)
        case llvm::Instruction::FAdd:   return "fadd";
        case llvm::Instruction::FSub:   return "fsub";
        case llvm::Instruction::FMul:   return "fmul";
        case llvm::Instruction::FDiv:   return "fdiv";
        case llvm::Instruction::FRem:   return "frem";

            // Bitwise
        case llvm::Instruction::Shl:    return "shl";
        case llvm::Instruction::LShr:   return "lshr";
        case llvm::Instruction::AShr:   return "ashr";
        case llvm::Instruction::And:    return "and";
        case llvm::Instruction::Or:     return "or";
        case llvm::Instruction::Xor:    return "xor";

        default:     return "unknown";
    }
}

void printResult(){
    llvm::errs()<<"Function saved: "<<saveFuncs<<"\n";
    llvm::errs()<<"Unary operation counts:\n";
    for(const auto& p:unaryOpsCnt){
        llvm::errs()<<getUnaryOrBinaryOpName(p.first)<<"\t"<<p.second<<"\n";
    }
    llvm::errs()<<"\nBinary operation counts "<<binaryOpsCnt.size()<<":\n";
    for(const auto& p:binaryOpsCnt){
        llvm::errs()<<getUnaryOrBinaryOpName(p.first)<<"\t"<<p.second<<"\n";
    }
    llvm::errs()<<"\nIntrinsic operation counts "<<intrinsicCnt.size()<<":\n";
    for(const auto& p:intrinsicCnt){
        llvm::errs()<<llvm::Intrinsic::getBaseName(p.first)<<"\t"<<p.second<<"\n";
    }
}

int main(int argc, char **argv) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    llvm::InitLLVM X(argc, argv);
    llvm::EnableDebugBuffering = true;
    llvm::LLVMContext Context;

    std::string Usage =
            R"EOF(Alive-mutate, a stand-alone LLVM IR fuzzer cooperates  with Alive2. Alive2 version: )EOF";
    Usage += R"EOF(
see alive-mutate --help for more options,
)EOF";

    llvm::cl::HideUnrelatedOptions(mutatorArgs);
    llvm::cl::ParseCommandLineOptions(argc, argv, Usage);

    auto uni_M1 = openInputFile(Context, inputFile);
    std::shared_ptr M1 = std::move(uni_M1);
    if (!M1.get()) {
        llvm::errs() << "Could not read input file from '" << inputFile << "'\n";
        return -1;
    }

    if (outputFolder.back() != '/')
        outputFolder += '/';

    std::vector<int> patternSizeVec;
    for(int x:patternSize){
        patternSizeVec.push_back(x);
    }
    sort(patternSizeVec.begin(), patternSizeVec.end());
    //M1->dump();
    walkModule(M1, depth, patternSizeVec);
    printResult();
    return 0;
}

bool isValidOutputPath() {
    bool result = filesystem::status(string(outputFolder)).type() ==
                  filesystem::file_type::directory;
    return result;
}

std::string getOutputSrcFilename(int ith) {
    static filesystem::path inputPath = filesystem::path(string(inputFile));
    static string templateName = string(outputFolder) + inputPath.stem().string();
    return templateName + to_string(ith) + ".ll";
}
