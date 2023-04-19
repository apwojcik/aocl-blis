##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources(${TARGET_NAME}
    PRIVATE
        bla_gemm_check.h
        bla_gemmt_check.h
        bla_gemv_check.h
        bla_ger_check.h
        bla_hemm_check.h
        bla_hemv_check.h
        bla_her2_check.h
        bla_her2k_check.h
        bla_her_check.h
        bla_herk_check.h
        bla_symm_check.h
        bla_symv_check.h
        bla_syr2_check.h
        bla_syr2k_check.h
        bla_syr_check.h
        bla_syrk_check.h
        bla_trmm_check.h
        bla_trmv_check.h
        bla_trsm_check.h
        bla_trsv_check.h
        bla_gemm3m_check.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
