from typing import Dict, List

TIME_FORMAT: Dict[str, str] = {
    "start_time": "%Y%m%d%H%M%S",
    "open_datetime": "%Y%m%d%H%M%S",
}
DTAETIME_COLUMNS_TYPE: Dict[str, str] = {key: "str" for key in TIME_FORMAT}
DTAETIME_COLUMNS: List[str] = list(TIME_FORMAT.keys())

NUMERIC_COLUMNS_TYPE: Dict[str, str] = {
    "call_duration": "int32",
    "cfee": "int32",
    "lfee": "int32",
    "hour": "int8",
}
NUMERIC_COLUMNS: List[str] = list(NUMERIC_COLUMNS_TYPE.keys())

AREA_CODE_COLUMNS: List[str] = [
    "home_area_code",
    "visit_area_code",
    "called_home_code",
    "called_code",
]
AREA_CODE_COLUMNS_TYPE: Dict[str, str] = {key: "str" for key in AREA_CODE_COLUMNS}

CITY_COLUMNS: List[str] = ["phone1_loc_city", "phone2_loc_city"]
CITY_COLUMNS_TYPE: Dict[str, str] = {key: "str" for key in CITY_COLUMNS}

PROVINCE_COLUMNS: List[str] = ["phone1_loc_province", "phone2_loc_province"]
PROVINCE_COLUMNS_TYPE: Dict[str, str] = {key: "str" for key in PROVINCE_COLUMNS}

A_PRODUCT_ID_COLUMNS: List[str] = ["a_product_id"]
A_PRODUCT_ID_COLUMNS_TYPE: Dict[str, str] = {key: "str" for key in A_PRODUCT_ID_COLUMNS}

CATEGORICAL_COLUMNS: List[str] = [
    "a_serv_type",
    "long_type1",
    "roam_type",
    "dayofweek",
    "phone1_type",
    "phone2_type",
]
CATEGORICAL_COLUMNS_TYPE: Dict[str, str] = {
    key: "category" for key in CATEGORICAL_COLUMNS
}
