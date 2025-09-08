#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/ValueTracking.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <filesystem>

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


    llvm::cl::opt<bool>
            collectInstructions(
    "collect-instructions",
    llvm::cl::value_desc("collect instructions"),
    llvm::cl::desc("Collect and count instructions"),
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

void updateCntMap(std::unordered_map<unsigned, unsigned>& umap, unsigned opCode){
    if(umap.find(opCode)==umap.end()){
        umap.emplace(opCode, 0);
    }
    umap[opCode]+=1;
}

unsigned totalBits, totalKB;

void countKB(llvm::Instruction* inst, const llvm::DataLayout& layout){
    auto bitwidth = inst->getType()->getIntegerBitWidth();
    llvm::KnownBits kb(bitwidth);
    llvm::computeKnownBits(inst, kb,layout);
    auto bits = kb.getBitWidth();
    auto knownBits = (~(kb.Zero | kb.One)).popcount();
    totalBits+=bits;
    totalKB+=(bits-knownBits);
}

void walkKBModule(std::shared_ptr<llvm::Module> module){
    for(llvm::Function& func:*module){
        for(auto it=llvm::inst_begin(func), end_it = llvm::inst_end(func);it!=end_it;++it){
            if(it->getType()->isIntOrIntVectorTy()){
                countKB(&*it, module->getDataLayout());
            }
        }
    }
}

void walkModule(std::shared_ptr<llvm::Module> module){
    for(llvm::Function& func:*module){
        for(auto it=llvm::inst_begin(func), end_it = llvm::inst_end(func);it!=end_it;++it){
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
    llvm::errs()<<"Total bits: "<<totalBits<<"\nTotal Known Bits: "<<totalKB<<"\n";
    if(!collectInstructions)return;
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
    if(collectInstructions){
        walkModule(M1);
    }
    walkKBModule(M1);
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
