##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
     PRIVATE
          bli_her2.h
          bli_her2_unb_var1.c
          bli_her2_unb_var2.c
          bli_her2_unb_var3.c
          bli_her2_unb_var4.c
          bli_her2_var_oapi.c
          bli_her2_var.h
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
               bli_her2_unf_var1_amd.c
               bli_her2_unf_var4_amd.c
          )
else()
     target_sources(${TARGET_NAME}
          PRIVATE
               bli_her2_unf_var1.c
               bli_her2_unf_var4.c
          )
endif()