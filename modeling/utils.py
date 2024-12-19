import pandas as pd
import numpy as np


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("warmup epochs ", warmup_epochs, "warmup iterations ", warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def process_address(b):
    return ", ".join([_.strip(", ") for _ in b.split("    ") if len(_.strip(" ")) > 0])


def get_lon_lat(lon_str, lat_str):
    try:
        return float(lon_str), float(lat_str)
    except:
        return np.nan, np.nan

def parse_built_Sqft(content):
    try: 
        building_sqft = content["house_features"]["Building Sqft.:"]
    except:
        building_sqft = None
    if building_sqft is None:
        return np.nan
    built_sqft = str(building_sqft)
    try:
        built_sqft = built_sqft.split("(m²)")[0]
        built_sqft_, built_sqm_ = built_sqft[:-3], built_sqft[-3:]
        built_sqft_ = float(built_sqft_.replace(",", ""))
        built_sqm_ = float(built_sqm_)
        built_sqft_to_sqm = built_sqft_ * 0.092903 
        if abs(built_sqft_to_sqm - built_sqm_) / built_sqm_ < 0.05:
            return built_sqft_
        else:
            built_sqft_, built_sqm_ = built_sqft[:-2], built_sqft[-2:]
            built_sqft_ = float(built_sqft_.replace(",", ""))
            built_sqm_ = float(built_sqm_)
            built_sqft_to_sqm = built_sqft_ * 0.092903 
            if abs(built_sqft_to_sqm - built_sqm_) / built_sqm_ < 0.05:
                return built_sqft_
            else:
                return np.nan
    except Exception:
        return np.nan
    

def parse_ls1(ls1):
    ls1 = str(ls1)
    if ls1 == "nan":
        return np.nan
    if "Sqft." in ls1:
        ls_ = float(ls1.split("Sqft.")[1].replace("(m²)", "").replace(",", ""))
    if "Acres" in ls1:
        ls_ = float(ls1.split("Acres")[1].replace("(m²)", "").replace(",", ""))
    return ls_


def parse_lot_size(content):
    try: 
        lot_size = content["house_features"]["Lot Size:"]
    except:
        lot_size = "nan"
    lot_size = str(lot_size)
    if lot_size == "nan":
        return np.nan, ""
    ls1, ls2 = lot_size.split("/")
    lot_size_m2 = parse_ls1(ls1)
    return lot_size_m2, ls2


def parse_built_year(content):
    try: 
        built_year = str(content["house_features"]["Year Built:"])
    except:
        built_year = "nan"

    if built_year == "nan":
        return np.nan, np.nan
    year, source = built_year.split("/")
    year = int(year)
    if year == 20119:
        year = 2019
    return year, source.strip()

def get_garage_num(content):
    try:
        garage_string = content["house_features"]["Garage(s):"]
    except:
        garage_string = None
    if garage_string is None:
        return np.nan
    
    try:
        garage_num = float(garage_string.split("/")[0]) if "/" in garage_string else 0
        return garage_num
    except:
        return np.nan

def parse_bedroom(content):
    try:
        bedrooms = content["house_features"]["Bedrooms:"]
    except:
        bedrooms = np.nan  
    bedrooms = str(bedrooms)
    if bedrooms == "nan":
        return np.nan
    if "-" in bedrooms:
        bedrooms = [float(tem_) for tem_ in bedrooms.replace("Bedroom(s)", "").split("-")]
        if bedrooms[1] > 100:
            return bedrooms[0]
        else:
            return np.mean(bedrooms)
    else:
        return float(bedrooms.replace("Bedroom(s)", ""))

def parse_bath(content):

    try:
        baths = content["house_features"]["Baths:"]
    except:
        baths = np.nan  

    baths = str(baths)
    if baths == "nan":
        return np.nan
    baths = baths.replace(" Bath(s)", "")
    num_half, num_full = 0, 0
    if "Half" in baths:
        baths = baths.replace("Half", "").replace("Full", "")
        num_full, num_half = baths.split("&")
    else:
        num_half = 0
        num_full = baths.replace("Full", "")
    return float(num_full) + float(num_half) / 2.0 

def parse_maintenance_fee(content):
    try:
        mstr = content["house_features"]["Maintenance Fee:"]
    except:
        mstr = np.nan 
    mstr = str(mstr)
    mfee = np.nan
    if "No".lower() in mstr.lower():
        mfee = 0
    if "Yes".lower() in mstr.lower() and "/" not in mstr:
        mfee = np.nan
    if "/" in mstr.lower() and "Annually".lower() in mstr.lower() and "$" in mstr:
        mfee = float(mstr.split("/")[-2].replace("$", "").replace(",", "")) / 12
    
    if "/" in mstr.lower() and "Month".lower() in mstr.lower() and "$" in mstr:
        mfee = float(mstr.split("/")[-2].replace("$", "").replace(",", ""))
    
    if "/" in mstr.lower() and "Quarter".lower() in mstr.lower() and "$" in mstr:
        mfee = float(mstr.split("/")[-2].replace("$", "").replace(",", "")) / 3
    if mfee > 5000:
        mfee = np.nan
    return mfee


def extract_tax_rate_tax_table(tax_table):
    if len(tax_table) == 0:
        return np.nan
    for element_ in tax_table:
        for inner_element_ in element_:
            if "Total Tax Rate".lower() in inner_element_[0].lower():
                return round(float(inner_element_[1].replace("%", "").strip()), 3)
    return np.nan

def get_tax_rate(content):
    if "Tax Rate:" in content["house_features"]:
        return float(content["house_features"]["Tax Rate:"])
    else:
        return np.nan 

def get_recent_market_value(content):
    try:
        table1 = content["house_tax_table"][0]
        table_df = pd.DataFrame(table1[1:], columns=table1[0])
        recent_market_value = table_df.sort_values("Tax Year", ascending=False).iloc[0]["Market Value"]
        recent_market_value = eval(recent_market_value.replace("$", "").replace(",", ""))
        return recent_market_value
    except:
        return np.nan

def get_recent_tax_value(content):
    try:
        table1 = content["house_tax_table"][0]
        table_df = pd.DataFrame(table1[1:], columns=table1[0])
        recent_tax_value = table_df.sort_values("Tax Year", ascending=False).iloc[0]["Tax Assessment"]
        recent_tax_value = eval(recent_tax_value.replace("$", "").replace(",", ""))
        return recent_tax_value
    except:
        return np.nan


def parse_property_type(content):
    try: 
        property_type = content["house_features"]["Property Type:"]
    except:
        return None 

    property_type = str(property_type)
    if property_type == "Country Homes/Acreage - Free Standi":
        property_type = "Country Homes/Acreage"
    if "Multi-Family" in property_type:
        property_type = "Multi-Family"
    if "Single Family" in property_type:
        property_type = "Single Family"
    if "Single-Family" in property_type:
        property_type = "Single Family"
    if "townhouse" in property_type.lower() and "condo" in property_type.lower():
        property_type = "townhouse/condo"
    if "residential" in property_type.lower() and "mobile" in property_type.lower():
        property_type = "residential/mobile home"
    if "residential" in property_type.lower() and "condo" in property_type.lower():
        property_type = "residential/condo"
    if "residential" in property_type.lower() and "manufactured" in property_type.lower():
        property_type = "residential/manufactured"
    if "residential" in property_type.lower() and "townhouse" in property_type.lower():
        property_type = "townhouse"
    if "residential" in property_type.lower() and "lot" in property_type.lower():
        property_type = "residential/lot"
    if "country homes/acreage" in property_type.lower():
        property_type = "country homes/acreage"
    return property_type.lower()

def private_pool_feature(content):
    house_features = content["house_features"]
    if "Private Pool:" in house_features:
        return house_features["Private Pool:"]
    else:
        return None
    
def area_pool_feature(content):
    house_features = content["house_features"]
    if "Area Pool:" in house_features:
        return house_features["Area Pool:"]
    else:
        return None

def private_pool_desc_feature(content):
    house_features = content["house_features"]
    if "Private Pool Desc:" in house_features:
        return house_features["Private Pool Desc:"]
    else:
        return None
    
def private_pool_multiclass_feature(content):
    house_features = content["house_features"]
    if "Private Pool Desc:" in house_features:
        return [_.strip(" ,").lower() for _ in house_features["Private Pool Desc:"].split(",")]
    else:
        return []

def Foundation_multiclass_feature(content):
    house_features = content["house_features"]
    if "Foundation:" in house_features:
        return [_.strip(" ,") for _ in house_features["Foundation:"].split(",")]
    else:
        return []

def Garage_Types_multiclass_features(content):
    house_features = content["house_features"]
    if "Garage(s):" in house_features:
        try: 
            garage_str = house_features["Garage(s):"]
            return [_.strip(" ,") for _ in garage_str.split("/")[1].split(",") if len(_.strip(" ,")) > 0]
        except:
            return []
    else:
        return []

def Roof_Type_multiclass(content):
    house_features = content["house_features"]
    if "Roof:" in house_features:
        return [_.strip(" ,").lower() for _ in house_features["Roof:"].split(",")]
    else:
        return []

def floor_type_multiclass(content):
    house_features = content["house_features"]
    if "Floors:" in house_features:
        return [_.strip(" ,").lower() for _ in house_features["Floors:"].split(",")]
    else:
        return []
    
def exterior_type_multiclass(content):
    house_features = content["house_features"]
    exterior_type_set = set(['brick', 'brick veneer', 'cement board', 'concrete',
       'fiber cement', 'frame', 'hardiplank type', 'masonry ? all sides',
       'masonry ? partial', 'other', 'rock/stone', 'siding', 'stone',
       'stone veneer', 'stucco', 'vinyl', 'wood'])

    if "Exterior Type:" in house_features:
        res = [_.strip(" ,").lower() for _ in house_features["Exterior Type:"].split(",")]
        return [_ if _ in exterior_type_set else "other" for _ in res] 
    else:
        return []
    
def exterior_multiclass(content):
    house_features = content["house_features"]
    most_50_features = set(['back yard fenced',
    'covered patio/deck',
    'back yard',
    'sprinkler system',
    'patio/deck',
    'covered patio/porch',
    'rain gutters',
    'private yard',
    'porch',
    'fully fenced',
    'lighting',
    'balcony',
    'subdivision tennis court',
    'private driveway',
    'side yard',
    'gutters full',
    'gutters partial',
    'storage shed',
    'back green space',
    'outdoor kitchen',
    'spa/hot tub',
    'exterior steps',
    'storage',
    'controlled subdivision access',
    'partially fenced',
    'exterior gas connection',
    'balcony/terrace',
    'workshop',
    'outdoor living center',
    'gas grill',
    'outdoor grill',
    'no exterior steps',
    'fire pit',
    'private entrance',
    'garden',
    'garden(s)',
    'attached grill',
    'see remarks',
    'detached gar apt /quarters',
    'outdoor fireplace',
    'storm shutters',
    'pest tubes in walls',
    'screened porch',
    'controlled access',
    'courtyard',
    'fenced',
    'covered deck',
    'other',
    'barbecue',
    'artificial turf'])

    if "Exterior:" in house_features:
        return [_.strip(" ,").lower() for _ in house_features["Exterior:"].split(",") if _.strip(" ,").lower() in most_50_features]
    else:
        return []
    
def style_multiclass(content):
    house_features = content["house_features"]
    rename_dict = {"other style": "other"}
    style_set = set(['traditional',
                'contemporary/modern',
                'ranch',
                'other',
                'craftsman',
                'mediterranean',
                'colonial',
                'french'])
    if "Style:" in house_features:
        res = [_.strip(" ,").lower() for _ in house_features["Style:"].split(",")]
        res = [rename_dict[_] if _ in rename_dict else _ for _ in res]
        return [_  for _ in res if _ in style_set]
    else:
        return []

def finanace_option_multiclass(content):
    house_features = content["house_features"]
    rename_dict = {"cash": "cash sale", 
                   "cashs": "cash sale",
                   "seller to contribute": "seller to contribute",
                   "seller to contribute to buyer's closing costs": "seller to contribute",
                   "seller to contribute to buyer's cl": "seller to contribute",
                   "seller may contribute to buyer's closing c": "seller to contribute",
                   "seller may contribute to buyer's closing costs": "seller to contribute",
                   "texas veterans land board": "tvlb", 
                   "texas veterans land b": "tvlb",
                   "usdal": "usda loan",
                   "invst": "investor",
                   "ownfn": "owner financing",}
    if "Financing Considered:" in house_features:
        res = [_.strip(" ,_").lower() for _ in house_features["Financing Considered:"].split(",")]
        res = [rename_dict[_] if _ in rename_dict else _ for _ in res]
        return res
    else:
        return []


def extract_high_school_name(school_dict_list):
    if len(school_dict_list) == 0:
        return "", "", np.nan, "", []
    for school_dict_ in school_dict_list:
        if "High School High" in school_dict_["school name"]:
            return school_dict_["school name"].replace("High School High", "High School"), \
                school_dict_["school grades"], school_dict_["school stars"], \
                school_dict_["school rate"], school_dict_["school attributes"]
    return "", "", np.nan, "", []


def extract_mid_school_name(school_dict_list):
    if len(school_dict_list) == 0:
        return "", "", np.nan, "", []
    for school_dict_ in school_dict_list:
        if "Middle School Middle" in school_dict_["school name"]:
            return school_dict_["school name"].replace("Middle School Middle", "Middle School"), \
                school_dict_["school grades"], school_dict_["school stars"], \
                school_dict_["school rate"], school_dict_["school attributes"]
    return "", "", np.nan, "", []


def extract_elemetary_school_name(school_dict_list):
    if len(school_dict_list) == 0:
        return "", "", np.nan, "", []
    for school_dict_ in school_dict_list:
        if "Elementary School Elementary" in school_dict_["school name"]:
            return school_dict_["school name"].replace("Elementary School Elementary", "Elementary School"), \
                school_dict_["school grades"], school_dict_["school stars"], \
                school_dict_["school rate"], school_dict_["school attributes"]
    return "", "", np.nan, "", []


def school_org(content):
    if "house_schools" in content:
        house_schools = content["house_schools"]
        return list(np.unique([_["school org"] for _ in house_schools if "school org" in _]))
    else:
        return []