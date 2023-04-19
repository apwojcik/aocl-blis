##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
	PRIVATE
		bli_gemv_zen_ref.c
		bli_her2_zen_int_4.c
		bli_gemv_zen_int_4.c
    )

# Select AMD specific sources for AMD configurations.
if(${TARGET_ARCH} STREQUAL zen OR 
   ${TARGET_ARCH} STREQUAL zen2 OR
   ${TARGET_ARCH} STREQUAL zen3 OR
   ${TARGET_ARCH} STREQUAL zen4 OR
   ${TARGET_ARCH} STREQUAL amdzen)
	target_sources(${TARGET_NAME}
    	PRIVATE
     		bli_her_zen_int_amd.c
     	)
endif()