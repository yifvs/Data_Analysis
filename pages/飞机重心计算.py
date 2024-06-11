import streamlit as st
import pandas as pd

# 读取Excel文件中指定的Sheet，并去除列名中的空格
def read_fuel_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# 获取力臂值的函数
def get_balance_arm(df, volume):
    closest_row = df.iloc[(df['Volume'] - volume).abs().argmin()]
    balance_arm = closest_row['BA']
    return balance_arm

# 计算新的重心位置的函数
def calculate_new_cg(dry_weight, initial_arm, fuel_weight, balance_arm, total_weight):
    new_cg = (dry_weight * initial_arm + fuel_weight * balance_arm) / total_weight
    return new_cg

# 计算MAC百分比的函数
def calculate_mac(balance_arm):
    leading_edge = 627.1
    mac_length = 155.8
    mac_percentage = ((balance_arm - leading_edge) * 100) / mac_length
    return mac_percentage

st.title('737NG飞机重心计算器 V1.1')

st.markdown("""
### 使用说明

* **飞机最新的称重报告查询网址**: https://tdms.hnatechnic.com/login.shtml
* **路径**：专项管理--->载重平衡控制管理--->单机数据查询

### """, unsafe_allow_html=True)

st.image("Weight Terms.jpg")

st.markdown("* **干使用空重（OEW）**：指除商务载重（旅客、行李及货物）和燃油外飞机做好执行航班前准备的空机重量，包含餐食、饮用水、机载资料等。")
st.markdown("---")

# 选择飞机型号
aircraft_type = st.selectbox("选择飞机型号", ["737NG", "737MAX"])

# 添加单位转换器
st.sidebar.title("单位换算器")

unit_types = {
    "Mass": {
        "Kilogram": 1,
        "Gram": 1000,
        "Pound": 2.20462,
        "Ounce": 35.274
    },
    "Length": {
        "Meter": 1,
        "Kilometer": 0.001,
        "Centimeter": 100,
        "Millimeter": 1000,
        "Mile": 0.000621371,
        "Yard": 1.09361,
        "Foot": 3.28084,
        "Inch": 39.3701
    },
    "Density": {
        "KG/L": 1,
        "Pound/Gallon": 8.3454  # 1 kg/L = 8.3454 lb/gal
    }
}

unit_type = st.sidebar.selectbox("选择单位类型", list(unit_types.keys()))
units = list(unit_types[unit_type].keys())
input_value = st.sidebar.number_input("输入值", value=1.0)
input_unit = st.sidebar.selectbox("输入单位", units)
output_unit = st.sidebar.selectbox("输出单位", units)

conversion_factor = unit_types[unit_type][output_unit] / unit_types[unit_type][input_unit]
converted_value = input_value * conversion_factor

st.sidebar.write(f"{input_value} {input_unit} = {converted_value:.2f} {output_unit}")

# 读取Excel文件中的数据
# file_path = 'fuel_balance_arm.xlsx'  # 上传的Excel文件路径

# 获取用户输入
dry_operating_weight = st.number_input("请输入飞机的干使用空重（KG）", min_value=0.0)
center_tank_fuel_weight = st.number_input("请输入中央油箱燃油的重量（KG）", min_value=0.0)
left_main_tank_fuel_weight = st.number_input("请输入左主燃油箱的重量（KG）", min_value=0.0)
right_main_tank_fuel_weight = st.number_input("请输入右主燃油箱的重量（KG）", min_value=0.0)
initial_cg = st.number_input("请输入初始重心（MAC%）", min_value=0.0)
initial_balance_arm = st.number_input("请输入初始力臂（BA）", min_value=0.0)
fuel_density = st.number_input("请输入燃油密度（磅/加仑）", min_value=0.0)

# 添加提交按钮
if st.button("提交"):
    if fuel_density == 0:
        st.error("燃油密度不能为零，请输入一个大于零的值。")
    else:
        # 读取Excel文件中的数据
        if aircraft_type == "737NG":
            file_path = 'fuel_balance_arm.xlsx'
        else:
            file_path = '737MAX_fuel_balance_arm.xlsx'
            
        sheet_names = {
            'left_right_us': 'tank1&2 US',
            'left_right_liters': 'tank1&2 L',
            'center_us': 'center tank US',
            'center_liters': 'center tank L'
        }

        # 选择适当的Sheet
        left_right_df = read_fuel_data(file_path, sheet_names['left_right_us'])
        center_df = read_fuel_data(file_path, sheet_names['center_us'])

        # 计算总重
        total_weight = dry_operating_weight + center_tank_fuel_weight + left_main_tank_fuel_weight + right_main_tank_fuel_weight

        # 将燃油重量转换为体积（加仑）
        center_tank_weight_pounds = center_tank_fuel_weight * 2.20462  # kg 转换为 磅
        left_main_tank_weight_pounds = left_main_tank_fuel_weight * 2.20462  # kg 转换为 磅
        right_main_tank_weight_pounds = right_main_tank_fuel_weight * 2.20462  # kg 转换为 磅
        center_tank_volume = center_tank_weight_pounds / fuel_density
        left_main_tank_volume = left_main_tank_weight_pounds / fuel_density
        right_main_tank_volume = right_main_tank_weight_pounds / fuel_density

        # 计算左右主油箱的总燃油体积
        main_tanks_total_volume = left_main_tank_volume + right_main_tank_volume
        # 获取对应的力臂值
        center_tank_balance_arm = get_balance_arm(center_df, center_tank_volume)
        main_tanks_balance_arm = get_balance_arm(left_right_df, main_tanks_total_volume)

        # 计算新的平衡臂
        new_balance_arm = (dry_operating_weight * initial_balance_arm +
                           main_tanks_balance_arm * (left_main_tank_fuel_weight + right_main_tank_fuel_weight) +
                           center_tank_balance_arm * center_tank_fuel_weight) / total_weight

        # 计算新的重心位置
        new_cg = calculate_new_cg(dry_operating_weight, initial_balance_arm,
                                  center_tank_fuel_weight + left_main_tank_fuel_weight + right_main_tank_fuel_weight,
                                  new_balance_arm, total_weight)

        # 计算新的MAC百分比
        mac_percentage = calculate_mac(new_balance_arm)

        # 展示结果
        st.write(f"总重: {total_weight:.2f} KG")
        st.write(f"新的平衡臂: {new_cg:.2f} 英寸")
        st.write(f"当前飞机重心: {mac_percentage:.2f}%")
