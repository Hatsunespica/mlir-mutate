# mlir-mutate

This repo includes a dialect-independent fuzzer on MLIR. 

It performs following syntactical mutations:

1. Reorder operand arguments
2. Replace a value with
    1. a function argument (possibily add a new one)
    2. another value in the context with the same type
  3. Shuffle contiguous but independent instructions
  4. Randomly move an instruction. If any value breaks SSA property, it would be fixed by Mutation 2


## Build
```
cmake -Bbuild -DMLIR_ROOT=/PathToLLVMProject/llvm-project/mlir -GNinja
cd build
ninja
```

## Run

NOTICE: **If you have your custom dialect, please print it in a generic form** like `opt yourTest.mlir -mlir-print-op-generic`

The user needs to specify two input, the first is the MLIR file, and the second is the folder of saving generated mutants.
```
./mlir-mutate ~/tmp3.mlir ./test -n 10 --saveAll
```
The command above generates 10 mutants with input `~/tmp3.mlir` under `./test` folder

Or you can specify the time you want to run 

```
./mlir-mutate ~/tmp3.mlir ./test -t 1 --saveAll
```
The command above keeps generating mutants with input `~/tmp3.mlir` under `./test` folder in one second.

You can also specify a random seed if you want. mlir-mutate would choose a different random seed as it starts. 
```
./mlir-mutate ~/tmp3.mlir ./test -n 10 --saveAll -s 1271475865
```

## Example

Suppose we can have the following file `tmp3.mlir` from CIRCT (`comb` and `hw` are not builtin dialect)
```
"builtin.module"() ({
  "func.func"() <{function_type = (i4, i4) -> i4, sym_name = "tmp"}> ({
  ^bb0(%arg0: i4, %arg1: i4):
    %0 = "hw.constant"() {value = -2 : i4} : () -> i4
    %1 = "comb.and"(%0, %arg0) <{twoState}> : (i4, i4) -> i4
    %2 = "comb.and"(%0, %arg1) <{twoState}> : (i4, i4) -> i4
    %3 = "comb.and"(%2, %1) <{twoState}> : (i4, i4) -> i4
    "func.return"(%3) : (i4) -> ()
  }) : () -> ()
}) : () -> ()
```

By running `./mlir-mutate ~/tmp3.mlir ./test -n 10 --saveAll`, we should see 10 mutants under `./test` file with different suffixes. 

Let's pick some:

```
//Current seed: 1271475865
module {
  func.func @tmp(%arg0: i4, %arg1: i4, %arg2: i4, %arg3: i4) -> i4 {
    %0 = "hw.constant"() {value = -2 : i4} : () -> i4
    %1 = "comb.and"(%arg2, %arg3) : (i4, i4) -> i4
    %2 = "comb.and"(%0, %arg0) : (i4, i4) -> i4
    %3 = "comb.and"(%0, %arg1) : (i4, i4) -> i4
    return %1 : i4
  }
}
```

The example above moves `comb.and"(%2, %1)` to the next of `how.constant` and both operands are repalced by new generated arguments.

```
//Current seed: 1271475865
module {
  func.func @tmp(%arg0: i4, %arg1: i4, %arg2: i4) -> i4 {
    %0 = "comb.and"(%arg0, %arg2) : (i4, i4) -> i4
    %1 = "hw.constant"() {value = -2 : i4} : () -> i4
    %2 = "comb.and"(%1, %arg1) : (i4, i4) -> i4
    %3 = "comb.and"(%2, %0) : (i4, i4) -> i4
    return %3 : i4
  }
}
```

The exmaple above moves another `comb.and` to the beginning and its operands are swapped and replaced by other value. 

## Last

Welcome to provide ideas and let me know if there are other dialect-independent mutations you want to add to the project!
