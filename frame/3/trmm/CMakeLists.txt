##Copyright (C) 2020-22, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_trmm.h
        bli_trmm_front.h
        bli_trmm_ll_ker_var2.c
        bli_trmm_lu_ker_var2.c
        bli_trmm_rl_ker_var2.c
        bli_trmm_ru_ker_var2.c
        bli_trmm_var.h
        bli_trmm_xx_ker_var2.c
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
               bli_trmm_front_amd.c
          )
else()
     target_sources(${TARGET_NAME}
          PRIVATE
               bli_trmm_front.c
          )
endif()

