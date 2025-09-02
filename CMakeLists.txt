cmake_minimum_required(VERSION 3.13.0)
project(mlir-mutate VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# CMake usually does decent job in finding out the proper dependency packages,
# so for most of the times you won't have to care about the <NAME>_ROOT variables.
# If you are not satisfied with the CMake's choices, you may use these variables
# to manually override them with the MLIR, Z3, or cvc5 you prefer.
set(MLIR_ROOT CACHE PATH "MLIR installation top-level directory")
option(USE_LIBC "Use libc++ in case the MLIR (and cvc5) is linked against libc++")

find_package(MLIR REQUIRED)
# MLIR_VERSION does not exist, so use LLVM_VERSION instead
# If package MLIR is found, package LLVM must have been found in the process
message(STATUS "Found MLIR ${LLVM_VERSION} from ${MLIR_DIR}/MLIRConfig.cmake")


add_subdirectory(tools)
