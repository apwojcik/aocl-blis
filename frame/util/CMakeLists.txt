##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_util_check.c
        bli_util_fpa.c
        bli_util_oapi.c
        bli_util_oapi_ba.c
        bli_util_oapi_ex.c
        bli_util_tapi.c
        bli_util_tapi_ba.c
        bli_util_tapi_ex.c
        bli_util_unb_var1.c
        bli_util_update.c
        bli_util_api_wrap.c
        bli_util_progress.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )