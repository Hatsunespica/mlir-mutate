#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MLIR_MUTATE_CAT("mlir-mutate-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
                                   llvm::cl::desc("first-mlir-file"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<string> output_folder("o",
                                    llvm::cl::desc("Specify output folder"),
                                    llvm::cl::value_desc("folder name"),
                                    llvm::cl::Optional);
llvm::cl::opt<bool>
        arg_verbose("verbose", llvm::cl::desc("Be verbose about what's going on"),
                    llvm::cl::Hidden, llvm::cl::init(false),
                    llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<bool>
        remove_repetition("remove_repetition", llvm::cl::desc("Remove repetitions in the result"),
                          llvm::cl::Hidden, llvm::cl::init(false),
                          llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<bool>
        slice_function("slice_function", llvm::cl::desc("Slice functions"),
                       llvm::cl::Hidden, llvm::cl::init(false),
                       llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<bool>
        repl("repl", llvm::cl::desc("Checking results in a repl way"),
             llvm::cl::Hidden, llvm::cl::init(false),
             llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<int>
        depth("depth", llvm::cl::desc("Depth of search when performing slicing"),
              llvm::cl::Hidden, llvm::cl::init(5), llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::list<std::string>
        cl_dialects("dialects", llvm::cl::desc("Slice operations within given dialects"));

llvm::cl::list<std::string>
        cl_operations("operations", llvm::cl::desc("Extra interesting operations"));

filesystem::path inputPath, outputPath;
llvm::StringMap<bool> interestingOperations;
llvm::StringSet<> interestingDialects;

bool isInteresting(mlir::Operation* op){
    auto opName = op->getName();
    auto opNameStringRef = opName.getStringRef();
    if(!interestingOperations.contains(opNameStringRef)){
        auto dialectName = opName.getDialect()->getNamespace();
        bool isInteresting = interestingDialects.contains(dialectName);
        interestingOperations.try_emplace(opNameStringRef, isInteresting);
    }
    return interestingOperations[opNameStringRef];
}

bool isValidInputPath();
void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited, int depth);
void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited);
void initInterestingOperationsAndDialects(){
    for(const auto& str: cl_dialects){
        interestingDialects.insert(llvm::StringRef(str));
    }
    for(const auto& str: cl_operations){
        interestingOperations.try_emplace(str, true);
    }
}

mlir::BlockArgument addParameter(mlir::func::FuncOp &func, mlir::Type ty) {
    auto result = func.insertArgument(func.getNumArguments(), ty, {}, func->getLoc());
    return func.getArgument(func.getNumArguments() - 1);
}

void addResult(mlir::func::FuncOp &func, mlir::Value val) {
    mlir::Operation &retOp = func.getFunctionBody().getBlocks().front().back();
    retOp.insertOperands(retOp.getNumOperands(), val);
}

mlir::func::FuncOp moveToFunc(MLIRContext &context,
                              std::vector<mlir::Operation *> ops,
                              mlir::Location loc) {

    mlir::OpBuilder builder(&context);

    mlir::FunctionType funcTy = mlir::FunctionType::get(&context, {}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "tmp", funcTy);
    mlir::Block *blk = func.addEntryBlock();

    auto retOp = builder.create<mlir::func::ReturnOp>(func->getLoc());
    blk->push_back(retOp.getOperation());

    unordered_set<mlir::Operation *> needReturn;
    unordered_map<mlir::Operation *, mlir::Operation *> um;
    // arg_num -> current arg_num;
    unordered_map<int, mlir::BlockArgument> arg_um;
    std::vector<mlir::Operation *> stk;

    for (auto op : ops) {
        mlir::Operation *cur = op->clone();
        stk.push_back(cur);
        um.insert({op, cur});
        needReturn.insert(cur);

        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            mlir::Value arg = op->getOperand(i);
            mlir::Type arg_ty = arg.getType();
            if (mlir::Operation *definingOp = arg.getDefiningOp(); definingOp) {
                if (auto it = um.find(definingOp); it != um.end()) {
                    /*
                     * Calc the result index in definingOp
                     * Assume there are multiple returns
                     */
                    size_t idx = 0;
                    needReturn.erase(it->second);
                    for (; idx < definingOp->getNumResults(); ++idx) {
                        if (definingOp->getResult(idx) == arg) {
                            cur->setOperand(i, it->second->getResult(idx));
                        }
                    }
                } else {
                    mlir::BlockArgument newArg = addParameter(func, arg_ty);
                    cur->setOperand(i, newArg);
                }
            } else {
                mlir::BlockArgument blk_arg = llvm::dyn_cast<mlir::BlockArgument>(arg);
                int arg_num = blk_arg.getArgNumber();
                if (arg_um.find(arg_num) == arg_um.end()) {
                    arg_um.insert({arg_num, addParameter(func, arg_ty)});
                }
                cur->setOperand(i, arg_um[arg_num]);
            }
        }
    }

    while (!stk.empty()) {
        blk->push_front(stk.back());
        stk.pop_back();
    }

    for (auto op : needReturn) {
        if (isInteresting(op)) {
            for (auto res_it = op->result_begin(); res_it != op->result_end();
                 ++res_it) {
                addResult(func, *res_it);
            }
        }
    }

    funcTy = mlir::FunctionType::get(&context, func.getArgumentTypes(),
                                     retOp.getOperation()->getOperandTypes());
    func.setFunctionType(funcTy);

    return func;
}

std::string funcToString(mlir::func::FuncOp func) {
    std::string result;
    llvm::raw_string_ostream os(result);
    func.print(os);
    return os.str();
}

std::vector<std::vector<mlir::Operation*>> sliceFunctions(ModuleOp moduleOp), extractFunctions(ModuleOp moduleOp);

int main(int argc, char *argv[]) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    llvm::PrettyStackTraceProgram X(argc, argv);
    llvm::EnableDebugBuffering = true;

    llvm::cl::ParseCommandLineOptions(argc, argv);

    DialectRegistry registry;

    // Register the standard passes we want.
    mlir::registerCSEPass();
    mlir::registerSCCPPass();
    mlir::registerInlinerPass();
    mlir::registerCanonicalizerPass();
    mlir::registerViewOpGraphPass();
    mlir::registerSymbolDCEPass();
    MLIRContext context(registry);
    mlir::registerAllDialects(context);
    context.appendDialectRegistry(registry);
    context.allowUnregisteredDialects();
    initInterestingOperationsAndDialects();

    if (!isValidInputPath()) {
        llvm::errs() << "Invalid input file!\n";
        return 1;
    }

    string errorMessage;
    auto src_file = openInputFile(filename_src, &errorMessage);

    if (!src_file) {
        llvm::errs() << errorMessage << "\n";
        return 66;
    }
    llvm::SourceMgr src_sourceMgr;
    ParserConfig parserConfig(&context);
    src_sourceMgr.AddNewSourceBuffer(move(src_file), llvm::SMLoc());
    auto ir_before = parseSourceFile<ModuleOp>(src_sourceMgr, parserConfig);
    ModuleOp moduleOp = ir_before.release();


    std::vector<std::vector<mlir::Operation*>> data;
    if(slice_function){
        data=sliceFunctions(moduleOp);
    }else{
        data= extractFunctions(moduleOp);
    }


    std::vector<pair<std::vector<mlir::Operation *>, int>> v;
    for (const auto &ele : data) {
        v.push_back({ele, ele.size()});
    }
    std::sort(v.begin(), v.end(),
              [](auto &a, auto &b) { return a.first > b.first; });

    std::vector<std::pair<mlir::func::FuncOp, int>> result;

    std::vector<mlir::func::FuncOp> funcs;
    for (size_t i = 0; i < v.size(); ++i) {
        funcs.push_back(moveToFunc(context, v[i].first, moduleOp.getLoc()));
    }

    if(remove_repetition){
        std::unordered_map<std::string, std::pair<mlir::func::FuncOp, int>> filter;
        for (int i = 0; i < funcs.size(); ++i) {
            std::string str = funcToString(funcs[i]);
            if (filter.find(str) == filter.end()) {
                filter.insert(std::make_pair(str, std::make_pair(funcs[i], 0)));
            }
            filter[str].second++;
        }
        for (const auto &p : filter) {
            result.push_back(p.second);
        }
    }else{
        assert(funcs.size()==v.size());
        for(int i=0;i<v.size();++i){
            result.push_back({funcs[i], v[i].second});
        }
    }

    sort(result.begin(), result.end(),
         [](std::pair<mlir::func::FuncOp, int> &a,
            std::pair<mlir::func::FuncOp, int> &b) {
             return a.second > b.second;
         });

    llvm::errs() << "Final result size: " << result.size();
    llvm::errs() << "\n";

    if (!output_folder.empty()) {
        llvm::errs() << "Start writing to files\n";

        auto destFolder = std::filesystem::path(std::string(output_folder));

        for (size_t i = 0; i < result.size(); ++i) {
            std::error_code ec;
            if (!std::filesystem::is_directory(destFolder)) {
                std::filesystem::create_directory(destFolder);
            }
            std::string outputFileName = destFolder.string();
            if (outputFileName.back() != '/') {
                outputFileName.push_back('/');
            }
            outputFileName += to_string(i) + ".mlir";
            llvm::raw_fd_ostream fout(outputFileName, ec);
            fout << "// " << result[i].second << "\n";
            result[i].first.print(fout);
            fout.close();
            llvm::errs() << "file wrote to " << outputFileName << "\n";
        }

        llvm::errs() << "Writing files done\n";
    }

    if(repl){
        int x;
        while (cin >> x && x != -1) {
            if (x > result.size()) {
                llvm::errs() << "out of range\n";
            } else {
                result[x].first.dump();
            }
        }
    }

    return 0;
}

std::vector<std::vector<mlir::Operation*>> sliceFunctions(ModuleOp moduleOp){
    std::unordered_set<mlir::Operation *> visited;
    std::vector<mlir::Operation *> tmp;
    std::vector<std::vector<mlir::Operation *>> data;
    int opCnt = 0;
    int argDepth = depth;
    moduleOp.walk([&visited, &data, &tmp, &opCnt,
                      &argDepth](mlir::Operation *op) {
        if (isInteresting(op)) {
            ++opCnt;
            if (visited.find(op) == visited.end()) {
                visit(op, tmp, visited, argDepth);
            }
            // We also consider DAGs with 1 operations
            if (tmp.size()) {
                data.push_back(tmp);
            }
            tmp.clear();
            // We clean visited to don't filter out operations
            visited.clear();
        }
    });

    llvm::errs() << "Sliced functions: " << data.size() << "\n";
    llvm::errs() << "Number of operations: " << opCnt << "\n";
    return data;
}

std::vector<std::vector<mlir::Operation*>> extractFunctions(ModuleOp moduleOp){
    std::unordered_set<mlir::Operation *> visited;
    std::vector<mlir::Operation *> tmp;
    std::vector<std::vector<mlir::Operation *>> data;
    int opCnt = 0;
    moduleOp.walk([&visited, &data, &tmp, &opCnt](mlir::Operation *op) {
        if (isInteresting(op)) {
            ++opCnt;
            if (visited.find(op) == visited.end()) {
                visit(op, tmp, visited);
            }
            // We also consider DAGs with 1 operations
            if (tmp.size()) {
                data.push_back(tmp);
            }
            tmp.clear();
        }
    });
    llvm::errs() << "Extracted functions: " << data.size() << "\n";
    llvm::errs() << "Number of operations: " << opCnt << "\n";
    return data;
}


bool isValidInputPath() {
    bool result = filesystem::status(string(filename_src)).type() ==
                  filesystem::file_type::regular;
    if (result) {
        inputPath = filesystem::path(string(filename_src));
    }
    return result;
}

void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited, int depth) {
    if (depth == 0) {
        return;
    }
    if (!isInteresting(op)) {
        return;
    }
    if (visited.find(op) == visited.end()) {
        visited.insert(op);
        for (Value operand : op->getOperands()) {
            if (Operation *producer = operand.getDefiningOp()) {
                visit(producer, tmp, visited, depth - 1);
            }
        }
        tmp.push_back(op);
    }
}

void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited) {
    if (!isInteresting(op)) {
        return;
    }
    if (visited.find(op) == visited.end()) {
        visited.insert(op);
        for (Value operand : op->getOperands()) {
            if (Operation *producer = operand.getDefiningOp()) {
                visit(producer, tmp, visited);
            }
        }
        tmp.push_back(op);
        mlir::OpResult result = op->getResult(0);
        for (Operation *userOp : result.getUsers()) {
            visit(userOp, tmp, visited);
        }
    }
}