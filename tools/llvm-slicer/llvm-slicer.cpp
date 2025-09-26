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


    llvm::cl::opt<bool>
            saveAll(
    "saveAll",
    llvm::cl::value_desc("save all mutants"),
    llvm::cl::desc("Save mutants to disk (default=false)"),
    llvm::cl::cat(mutatorArgs), llvm::cl::init(false)
    );


}

std::string getOutputSrcFilename(int ith);
bool isValidOutputPath();

llvm::StringSet<> invalidFunctions;
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
    auto functionType = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), args, false);
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
    auto returnInst = llvm::ReturnInst::Create(ctx, bb);
    return function;
}

void saveFunctionToFile(llvm::Function* func, const std::string &Path) {
    std::error_code EC;
    llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_Text);

    if (EC) {
        llvm::errs() << "Error opening file " << Path << ": " << EC.message() << "\n";
        return;
    }

    // Print only the function body (with definition).
    func->print(OS, nullptr);
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
           || isCallToNonIntrinsic(I) || llvm::isa<llvm::LoadInst>(I)
           || llvm::isa<llvm::AllocaInst>(I) || llvm::isa<llvm::StoreInst>(I)
           || !llvm::isa<llvm::Operator>(I)
           || llvm::isa<llvm::ICmpInst>(I) || llvm::isa<llvm::ZExtInst>(I)
           || llvm::isa<llvm::SExtInst>(I) || llvm::isa<llvm::TruncInst>(I)
           || llvm::isa<llvm::GetElementPtrInst>(I) || llvm::isa<llvm::SelectInst>(I);
}

void sliceInstruction(llvm::Instruction* inst, llvm::SmallVector<llvm::Instruction*>& insts, int depth){
    if(depth!=0&&!specialCheck(inst)){
        insts.push_back(inst);
        for(size_t i=0;i<inst->getNumOperands();++i){
            auto operand_inst = llvm::dyn_cast<llvm::Instruction>(inst->getOperand(i));
            if(operand_inst){
                sliceInstruction(operand_inst, insts, depth-1);
            }
        }
    }
}

void walkModule(std::shared_ptr<llvm::Module> module, int depth){
    bool shouldSlice = depth >1;
    llvm::SmallVector<llvm::Instruction*> insts;
    for(llvm::Function& func:*module){
        for(auto it=llvm::inst_begin(func), end_it = llvm::inst_end(func);it!=end_it;++it){
            //Slice on the current instruction
            if(shouldSlice && !it->isTerminator()&& !specialCheck(&*it)){
                sliceInstruction(&*it, insts, depth);
                if(insts.size()>1){
                    std::reverse(insts.begin(), insts.end());
                    auto func = moveToFunction(module->getContext(), insts);

                    auto outputPath = getOutputPath(func->getName().str());
                    saveFunctionToFile(func,outputPath);
                    ++saveFuncs;
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

    //M1->dump();
    walkModule(M1, depth);
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
