##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        f77_amax_sub.c
        f77_amin_sub.c
        f77_asum_sub.c
        f77_dot_sub.c
        f77_nrm2_sub.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )