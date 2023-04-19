##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
        PRIVATE
    bli_gemmbb_ref.c
    bli_gemmtrsmbb_ref.c
    bli_trsmbb_ref.c
    )

