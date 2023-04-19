##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

# Starting from CMake 3.10, the module FindOpenMP exposes language components.
cmake_minimum_required(VERSION 3.10.0)

project(AOCL-Blis C)

cmake_policy(SET CMP0076 NEW)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(AOCL_BLIS_FAMILY "zen" CACHE STRING "AOCL BLIS family name")
if ("${AOCL_BLIS_FAMILY}" STREQUAL "")
    message(FATAL_ERROR "Machine configuration missing! Select one of zen, zen2, zen3, zen4 or amdzen")
endif ()

set(TARGET_ARCH ${AOCL_BLIS_FAMILY})
set(TARGET_NAME blis)
set(AOCL_BLIS_ZEN TRUE)

if (${AOCL_BLIS_FAMILY} STREQUAL "auto")
  set(AUTO_CONFIG_PY "${CMAKE_CURRENT_SOURCE_DIR}/build/auto_config.py")
  # Run python script to find the architecture family name
  execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${AUTO_CONFIG_PY}
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Auto configuring the family :" ${CMD_OUTPUT})
  set(AOCL_BLIS_FAMILY ${CMD_OUTPUT})
endif ()

if(${AOCL_BLIS_FAMILY} STREQUAL "zen")
  add_definitions(-DBLIS_FAMILY_ZEN)
  add_definitions(-DBLIS_CONFIG_ZEN)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen2")
  add_definitions(-DBLIS_FAMILY_ZEN2)
  add_definitions(-DBLIS_CONFIG_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen3")
  add_definitions(-DBLIS_FAMILY_ZEN3)
  add_definitions(-DBLIS_CONFIG_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen4")
  add_definitions(-DBLIS_FAMILY_ZEN4)
  add_definitions(-DBLIS_CONFIG_ZEN4)
  add_definitions(-DBLIS_KERNELS_SKX)
  add_definitions(-DBLIS_KERNELS_ZEN4)
  add_definitions(-DBLIS_KERNELS_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
  set(AOCL_BLIS_ZEN FALSE)
  add_definitions(-DBLIS_FAMILY_AMDZEN)
  add_definitions(-DBLIS_CONFIG_ZEN4)
  add_definitions(-DBLIS_CONFIG_ZEN3)
  add_definitions(-DBLIS_CONFIG_ZEN2)
  add_definitions(-DBLIS_CONFIG_ZEN)
  add_definitions(-DBLIS_CONFIG_GENERIC)
  add_definitions(-DBLIS_KERNELS_SKX)
  add_definitions(-DBLIS_KERNELS_ZEN4)
  add_definitions(-DBLIS_KERNELS_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_HASWELL)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_GENERIC)
else ()
  message(FATAL_ERROR "Wrong machine configuration. Select one of zen, zen2, zen3, zen4 or amdzen")
endif ()

set(TARGET_ARCH ${AOCL_BLIS_FAMILY})
message(STATUS "AOCL_BLIS_FAMILY selected:${AOCL_BLIS_FAMILY}")

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_VERBOSE "Enable VERBOSE mode for build" OFF)
option(ENABLE_MULTITHREADING "Enable Multi threading" OFF)
option(ENABLE_OPENMP "Enable Openmp mode" OFF)
option(ENABLE_JRIR_SLAB "Request slab thread in jr and ir loops" ON)
option(ENABLE_JRIR_RR "Request round robin thread in jr and ir loops" OFF)
option(ENABLE_PBA_POOLS "Internal memory pools for packing blocks" ON)
option(ENABLE_SBA_POOLS "Internal memory pools for small blocks" ON)
option(ENABLE_MEM_TRACING "Memory tracing output" OFF)
option(ENABLE_BLAS "BLAS compatiblity layer" ON)
option(ENABLE_CBLAS "CBLAS compatiblity layer" ON)
option(ENABLE_MIXED_DT "Mixed datatype" ON)
option(ENABLE_MIXED_DT_EXTRA_MEM "Mixed datatype optimization requiring extra memory" ON)
option(ENABLE_SUP_HANDLING "Small matrix handling" ON)
option(ENABLE_MEMKIND "libmemkind for manage memory pools" OFF)
option(ENABLE_PRAGMA_OMP_SIMD "pragma openmp simd" ON)
option(ENABLE_SANDBOX "Sandbox implementation for gemm" OFF)
option(BLIS_ENABLE_ILP64 "ENABLE BLIS ILP64" OFF)
option(ENABLE_INT_TYPE_SIZE " Internal BLIS integers ,used in native BLIS interfaces based on architecture dependent " ON)
option(ENABLE_BLASTEST "Enable the blastest" OFF)
option(ENABLE_TESTCPP_TESTING "Enabling testcpp" OFF)
option (ENABLE_NO_UNDERSCORE_API "export APIs without underscore" OFF)
option (ENABLE_UPPERCASE_API "export APIs with uppercase" OFF)
option (ENABLE_COMPLEX_RETURN_INTEL "Enable complex_return_intel" OFF)
option (ENABLE_TRSM_PREINVERSION "Enable TRSM preinversion" ON)
option (ENABLE_AOCL_DYNAMIC "Enable Dynamic Multi-threading" OFF)
option(DISABLE_BLIS_ARCH_TYPE "Disable BLIS_ARCH_TYPE functionality" OFF)
option(RENAME_BLIS_ARCH_TYPE "Rename BLIS_ARCH_TYPE env var renamed to supplied value" BLIS_ARCH_TYPE)
option(BUILD_TESTING "Build tests" OFF)

if(ENABLE_OPENMP)
    set(ENABLE_MULTITHREADING ON CACHE INTERNAL "Force multithreading for OpenMP")
    find_package(OpenMP REQUIRED COMPONENTS C)
endif()

find_package(PythonInterp 3 REQUIRED)

if(${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
  set(REF_KERNEL_MIRRORING_PY ${CMAKE_CURRENT_SOURCE_DIR}/build/blis_ref_kernel_mirror.py)
  message(STATUS "ref_kernel mirroring for fat binary")
  # Run python script to find the architecture family name
  execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${REF_KERNEL_MIRRORING_PY} ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Ref Kernel Mirroring :" ${CMD_OUTPUT})
endif()

if(ENABLE_NO_UNDERSCORE_API)
    add_definitions(-DBLIS_ENABLE_NO_UNDERSCORE_API)
endif()

if(ENABLE_COMPLEX_RETURN_INTEL)
    set(BLIS_ENABLE_COMPLEX_RETURN_INTEL TRUE)
else()
    set(BLIS_DISABLE_COMPLEX_RETURN_INTEL TRUE)
endif()

if(ENABLE_UPPERCASE_API)
     add_definitions(-DBLIS_ENABLE_UPPERCASE_API)
endif()

if(ENABLE_AOCL_DYNAMIC)
     set(AOCL_DYNAMIC TRUE)
endif()

if(BUILD_SHARED_LIBS)
    set(BLIS_ENABLE_SHARED TRUE)
    if(ENABLE_BLASTEST)
        add_definitions(-DAOCL_SUPPORT_BLASTEST_FOR_SHARED)
    endif()
endif ()

# Enable LP64/ILP64
if (BLIS_ENABLE_ILP64)
  set(BLIS_BLAS_INT_TYPE_SIZE TRUE)
  set(BLAS_INT_TYPE_SIZE "64")
  add_definitions(-DF2C_ENABLE_ILP64)
else ()
  set(BLIS_BLAS_INT_TYPE_SIZE TRUE)
  set(BLAS_INT_TYPE_SIZE "32")
endif ()

if(ENABLE_TRSM_PREINVERSION)
  set(BLIS_ENABLE_TRSM_PREINVERSION TRUE)
else()
  add_definitions(-DBLIS_DISABLE_TRSM_PREINVERSION)
endif()

if(ENABLE_INT_TYPE_SIZE)
  set(BLIS_INT_TYPE_SIZE TRUE)
  set(INT_TYPE_SIZE "64")
else()
  set(BLIS_INT_TYPE_SIZE TRUE)
  set(INT_TYPE_SIZE "32")
endif()

if(BLIS_ENABLE_ILP64 AND NOT ENABLE_INT_TYPE_SIZE)
    message(FATAL_ERROR "for ILP64 we must enable ENABLE_INT_TYPE_SIZE with BLIS_INT_TYPE_SIZE = 64 ")
endif()

if(ENABLE_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
endif()

if(ENABLE_JRIR_RR)
    message(STATUS "Round robin thread method enabled")
    set(BLIS_ENABLE_JRIR_RR TRUE)
elseif(ENABLE_JRIR_SLAB)
    message(STATUS "SLAB thread method enabled")
    set(BLIS_ENABLE_JRIR_SLAB TRUE)
else()
    message(WARNING "Unsupported method of thread partitioning in jr and ir loops")
endif()

if(ENABLE_PBA_POOLS)
    set(BLIS_ENABLE_PBA_POOLS TRUE)
endif()

if(ENABLE_SBA_POOLS)
    set(BLIS_ENABLE_SBA_POOLS TRUE)
endif()

if(ENABLE_MEM_TRACING)
    set(BLIS_ENABLE_MEM_TRACING FALSE)
endif()

if(ENABLE_BLAS)
    add_definitions(-DBLIS_ENABLE_BLAS)
else()
    add_definitions(-DBLIS_DISABLE_BLAS)
endif()

if(ENABLE_CBLAS)
    add_definitions(-DBLIS_ENABLE_CBLAS)
    if(NOT ENABLE_BLAS)
        # Force BLAS layer when CBLAS is enabled
        add_definitions(-DBLIS_ENABLE_BLAS)
    endif()
else()
    add_definitions(-DBLIS_DISABLE_CBLAS)
endif()

if(ENABLE_BLASTEST)
    add_definitions(-DBLIS_ENABLE_BLAS)
    add_definitions(-DBLIS_ENABLE_CBLAS)
endif()

if(ENABLE_TESTCPP_TESTING)
  add_definitions(-DBLIS_ENABLE_BLAS)
  add_definitions(-DBLIS_ENABLE_CBLAS)
endif()

if(ENABLE_MIXED_DT)
    set(BLIS_ENABLE_MIXED_DT TRUE)
endif()

if(ENABLE_MIXED_DT_EXTRA_MEM)
    set(BLIS_ENABLE_MIXED_DT_EXTRA_MEM TRUE)
endif()

if(ENABLE_SUP_HANDLING)
    set(BLIS_ENABLE_SUP_HANDLING TRUE)
endif()

if(ENABLE_MEMKIND)
    set(BLIS_ENABLE_MEMKIND FALSE)
endif()

if(ENABLE_PRAGMA_OMP_SIMD)
    set(BLIS_ENABLE_PRAGMA_OMP_SIMD TRUE)
endif()

if(ENABLE_SANDBOX)
    set(BLIS_ENABLE_SANDBOX FALSE)
endif()

if(CMAKE_C_COMPILER_ID STREQUAL MSVC)
    include_directories(${PROJECT_SOURCE_DIR}/external/msvc)
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /MD ")
    if(NOT BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /MT ")
    endif()
endif()

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

if(NOT BUILD_SHARED_LIBS)
    add_definitions(-DBLIS_IS_BUILDING_LIBRARY)
endif()

if(ENABLE_MULTITHREADING)
  set(LIB_NAME "${TARGET_NAME}-mt")
else()
  set(LIB_NAME ${TARGET_NAME})
endif()

if(DISABLE_BLIS_ARCH_TYPE)
    set(BLIS_DISABLE_BLIS_ARCH_TYPE TRUE)
else()
    set(BLIS_DISABLE_BLIS_ARCH_TYPE FALSE)
endif()

if(RENAME_BLIS_ARCH_TYPE)
    set(__blis_arch_type_name TRUE)
    set(rename_blis_arch_type "${RENAME_BLIS_ARCH_TYPE}")
else()
    set(__blis_arch_type_name TRUE)
    set(rename_blis_arch_type "BLIS_ARCH_TYPE")
endif()

if(ENABLE_VERBOSE)
    message("---cmake configurations---")
    message(CMAKE_C_COMPILER_ID : ${CMAKE_C_COMPILER_ID})
    message(CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE})
    message(BLIS_ENABLE_JRIR_SLAB : ${BLIS_ENABLE_JRIR_SLAB})
    message(BLIS_ENABLE_JRIR_RR : ${BLIS_ENABLE_JRIR_RR})
    message(BLIS_ENABLE_PBA_POOLS : ${BLIS_ENABLE_PBA_POOLS})
    message(BLIS_ENABLE_SBA_POOLS : ${BLIS_ENABLE_SBA_POOLS})
    message(BLIS_ENABLE_MEM_TRACING : ${BLIS_ENABLE_MEM_TRACING})
    message(BLIS_INT_TYPE_SIZE : ${BLIS_INT_TYPE_SIZE})
    message(BLIS_BLAS_INT_TYPE_SIZE : ${BLIS_BLAS_INT_TYPE_SIZE})
    message(BLIS_ENABLE_BLAS : ${BLIS_ENABLE_BLAS})
    message(BLIS_ENABLE_CBLAS : ${BLIS_ENABLE_CBLAS})
    message(BLIS_ENABLE_MIXED_DT : ${BLIS_ENABLE_MIXED_DT})
    message(BLIS_ENABLE_MIXED_DT_EXTRA_MEM : ${BLIS_ENABLE_MIXED_DT_EXTRA_MEM})
    message(BLIS_ENABLE_SUP_HANDLING : ${BLIS_ENABLE_SUP_HANDLING})
    message(BLIS_ENABLE_MEMKIND : ${BLIS_ENABLE_MEMKIND})
    message(BLIS_ENABLE_PRAGMA_OMP_SIMD : ${BLIS_ENABLE_PRAGMA_OMP_SIMD})
    message(BLIS_ENABLE_SANDBOX : ${BLIS_ENABLE_SANDBOX})
    message(BLIS_ENABLE_SHARED : ${BLIS_ENABLE_SHARED})
    message(DISABLE_BLIS_ARCH_TYPE : ${DISABLE_BLIS_ARCH_TYPE})
    message(RENAME_BLIS_ARCH_TYPE : ${RENAME_BLIS_ARCH_TYPE})
    message(BUILD_TESTING: ${BUILD_TESTING})
endif()

set(ENABLE_SIMD_FLAGS "AVX2" CACHE STRING "Set compiler SIMD flags")
set_property(CACHE ENABLE_SIMD_FLAGS PROPERTY STRINGS none SSE2 AVX AVX2)

if(CMAKE_C_COMPILER_ID STREQUAL MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0 /Oi /MP")
    set(INTR_GENERAL_LINK_FLAGS "${INTR_GENERAL_LINK_FLAGS} /RELEGE")
endif()

if (NOT ${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
    add_definitions(-DBLIS_CNAME=${TARGET_ARCH})
endif()

# Generate the bli_config.h header file
configure_file (build/bli_win_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/bli_config.h @ONLY)

string(APPEND HEADER_PATH
if(${AOCL_BLIS_FAMILY} STREQUAL "zen")
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/haswell/"
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen2")
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/haswell/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/amdzen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen4/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/generic/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/haswell/"
endif ()
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/0/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/0/copysc/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1d/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1f/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1m/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1m/packm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1m/unpackm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/gemv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/ger/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/hemv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/her/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/her2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/symv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/syr/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/syr2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/trmv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/trsv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/gemm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/gemm/ind/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/gemmt/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/hemm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/her2k/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/herk/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/symm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/syr2k/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/syrk/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/trmm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/trmm3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/trsm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/cast/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/check/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/noopt/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/proj/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/f77_sub/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/src/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/check/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/f2c/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/f2c/util/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/1e/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/1m/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/1r/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/bb/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/io/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/ri/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/ri3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/rih/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/ro/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/rpi/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/cntx/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/oapi/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/tapi/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/ukernels/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/thread/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/util/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/aocl_dtl/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/"
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${TARGET_ARCH})

# Flatten header python script file which expand header contents in blis.h
set(FLATTEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/build/flatten-headers.py")
set(BLIS_H "blis.h")

# Arguements for python script
set(C_COMMENT "-c")
set(VERBOSE "-v1")
set(INPUT "${CMAKE_CURRENT_SOURCE_DIR}/frame/include/${BLIS_H}")
set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/${TARGET_ARCH}/${BLIS_H}")
set(TEMP_DIR "${INCLUDE}")
set(DIR_H_PATH "${HEADER_PATH}")

# Run python script to generate monolithic header at configuration time
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${FLATTEN_PY} "${C_COMMENT}" "${VERBOSE}" "${INPUT}" "${OUTPUT}" "${TEMP_DIR}" "${DIR_H_PATH}"
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT)
message(STATUS "Generating monolithic header file :" ${CMD_OUTPUT})

# Logic to generate the cblas.h in include folder.
set(CBLAS_H "cblas.h")
# Arguements for python script
set(C_COMMENT "-c")
set(VERBOSE "-v1")
set(INPUT "${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/src/${CBLAS_H}")
set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/${TARGET_ARCH}/${CBLAS_H}")
set(TEMP_DIR "${INCLUDE}")
set(DIR_H_PATH "${HEADER_PATH}")

# Run python script to generate monolithic header at configuration time
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${FLATTEN_PY} "${C_COMMENT}" "${VERBOSE}" "${INPUT}" "${OUTPUT}" "${TEMP_DIR}" "${DIR_H_PATH}"
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT)
message(STATUS "Generating monolithic cblas header file :" ${CMD_OUTPUT})

# setting the blis version string
file (STRINGS "version" BLIS_VERSION)
set(BLIS_VERSION_STRING ${BLIS_VERSION})
string(TIMESTAMP BUILD_DATE "%Y%m%d")
add_definitions(-DBLIS_VERSION_STRING="AOCL-BLIS ${BLIS_VERSION_STRING} Build ${BUILD_DATE}")

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} SHARED
            ${CMAKE_CURRENT_BINARY_DIR}/bli_config.h
            ${CMAKE_CURRENT_BINARY_DIR}/include/${TARGET_ARCH}/blis.h)
    target_compile_definitions(${TARGET_NAME} PRIVATE -DBLIS_IS_BUILDING_LIBRARY)
    if(ENABLE_OPENMP)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_C)
    endif()
else()
    add_library(${TARGET_NAME} STATIC
            ${CMAKE_CURRENT_BINARY_DIR}/bli_config.h
            ${CMAKE_CURRENT_BINARY_DIR}/include/${TARGET_ARCH}/blis.h)
    if(ENABLE_OPENMP)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_C)
    endif()
endif()

set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE C OUTPUT_NAME ${LIB_NAME})

if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL Clang)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -fms-compatibility
            -fms-extensions
            -fdelayed-template-parsing)
endif()

if(${ENABLE_SIMD_FLAGS} MATCHES "AVX2")
    target_compile_options(${TARGET_NAME}
        PRIVATE
            $<$<C_COMPILER_ID:Clang,GNU>:-mavx2>
            $<$<C_COMPILER_ID:MSVC>:/arch:AVX2>
        )
elseif(${ENABLE_SIMD_FLAGS} MATCHES "AVX")
    target_compile_options(${TARGET_NAME}
        PRIVATE
            $<$<C_COMPILER_ID:Clang,GNU>:-mavx>
            $<$<C_COMPILER_ID:MSVC>:/arch:AVX>
        )
elseif(${ENABLE_SIMD_FLAGS} MATCHES "SSE2")
    target_compile_options(${TARGET_NAME}
        PRIVATE
            $<$<C_COMPILER_ID:Clang,GNU>:-msse2>
            $<$<C_COMPILER_ID:MSVC>:/arch:SSE2>
        )
endif()

if(${TARGET_ARCH} STREQUAL zen4 OR ${TARGET_ARCH} STREQUAL amdzen)
    if(CMAKE_C_COMPILER_ID STREQUAL MSVC)
        set(__compile_option /arch:AVX512)
    else()
        set(__compile_option "-mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl")
    endif()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen4/1/bli_amaxv_zen_int_avx512.c
            PROPERTIES COMPILE_FLAGS ${__compile_option})
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen4/3/bli_gemmtrsm_l_zen_16x14.c
            PROPERTIES COMPILE_FLAGS ${__compile_option})
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen4/3/bli_gemmtrsm_u_zen_16x14.c
            PROPERTIES COMPILE_FLAGS ${__compile_option})
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/kernels/skx/3/bli_dgemm_skx_asm_16x14.c
            PROPERTIES COMPILE_FLAGS ${__compile_option})
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/kernels/skx/3/bli_sgemm_skx_asm_32x12_l2.c
            PROPERTIES COMPILE_FLAGS ${__compile_option})
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DEXPMODULE)

add_subdirectory(config)
add_subdirectory(ref_kernels)
add_subdirectory(kernels)
add_subdirectory(frame)
add_subdirectory(aocl_dtl)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
    add_subdirectory(testsuite)
    add_subdirectory(bench)
    if(ENABLE_TESTCPP_TESTING)
        add_subdirectory(vendor/testcpp)
    endif()
    if (ENABLE_BLASTEST)
        add_subdirectory(blastest)
    endif()
endif()
