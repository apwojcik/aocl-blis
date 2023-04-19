##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
        PRIVATE
    bli_axpy2v_ref.c
    bli_axpyf_ref.c
    bli_dotaxpyv_ref.c
    bli_dotxaxpyf_ref.c
    bli_dotxf_ref.c
    )

