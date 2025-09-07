include(AddLLVM)

# /============================================================/
# 1. Build object files to check warnings/errors before linking
# /============================================================/

set(PROJECT_OBJ "llvm-kb-analyzer-obj")
add_library(${PROJECT_OBJ} OBJECT
    ./utils.cpp)

# Check MLIR and LLVM headers and include
target_include_directories(${PROJECT_OBJ} PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(${PROJECT_OBJ} PUBLIC ${MLIR_INCLUDE_DIRS})

# Warn about unused variables
target_compile_options(${PROJECT_OBJ} PUBLIC -Wunused-variable)
# Using cl::opt requires this
target_compile_options(${PROJECT_OBJ} PUBLIC -fno-rtti)
llvm_map_components_to_libnames(llvm_libs support core irreader analysis passes transformutils scalaropts bitwriter)

# Try using libc if possible
if(USE_LIBC)
    target_compile_options(${PROJECT_OBJ} PUBLIC -stdlib=libc++)
endif()

# /============================================================/
# 2. Build libllvm-kb-analyzer
# /============================================================/

set(PROJECT_LIB "llvmkbanalyzer")
add_library(${PROJECT_LIB})
target_link_libraries(${PROJECT_LIB} PUBLIC ${PROJECT_OBJ})

target_link_libraries(${PROJECT_LIB} PUBLIC ${LIB_LIST} pthread m curses ${llvm_libs})
llvm_update_compile_flags(${PROJECT_LIB})

# Try using libc if possible
if(USE_LIBC)
    target_link_options(${PROJECT_LIB} PUBLIC -stdlib=libc++)
endif()

# /============================================================/
# 3. Build executable
# /============================================================/

# Build executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(llvm-kb-analyzer llvm-kb-analyzer.cpp)
target_link_libraries(llvm-kb-analyzer ${PROJECT_LIB})


