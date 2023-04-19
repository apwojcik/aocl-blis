##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
     PRIVATE
          bli_gemv.h
          bli_gemv_unb_var1.c
          bli_gemv_unb_var2.c
          bli_gemv_unf_var2.c
          bli_gemv_var_oapi.c
          bli_gemv_var.h
     )

target_include_directories(${TARGET_NAME}
     PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
     )

# Select AMD specific sources for AMD configurations.
if(${TARGET_ARCH} STREQUAL zen OR 
   ${TARGET_ARCH} STREQUAL zen2 OR 
   ${TARGET_ARCH} STREQUAL zen3 OR 
   ${TARGET_ARCH} STREQUAL zen4 OR 
   ${TARGET_ARCH} STREQUAL amdzen)
     target_sources(${TARGET_NAME}
          PRIVATE
               bli_gemv_unf_var1_amd.c
          )
else()
     target_sources(${TARGET_NAME}
          PRIVATE
               bli_gemv_unf_var1.c
          )
endif()
