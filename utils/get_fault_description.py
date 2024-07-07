'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 17:42:06
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-07-04 23:01:56
FilePath: /Pipeline-Failure-Prediction/utils/get_fault_description.py
Description: 获取故障代码对应的故障描述
'''
def get_fault_description(fault_code):
    fault_dict = {
        1001: '物料推送装置故障1001',
        2001: '物料检测装置故障2001',
        4001: '填装装置检测故障4001',
        4002: '填装装置定位故障4002',
        4003: '填装装置填装故障4003',
        5001: '加盖装置定位故障5001',
        5002: '加盖装置加盖故障5002',
        6001: '拧盖装置定位故障6001',
        6002: '拧盖装置拧盖故障6002'
    }
    return fault_dict.get(fault_code, "未知故障代码")

# 示例使用
# fault_code = int(input("请输入故障代码："))
# print(get_fault_description(fault_code))