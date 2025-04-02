"""
An interface for Oyhand control https://www.oymotion.com/
"""


import os
import sys
print("\n".join(sys.path))

import numpy as np
import time

from pymodbus.client import ModbusSerialClient

upper_limits = [2.26, 100.21, 97.81, 101.37, 98.88, 89.98] # close
lower_limits = [36.76, 178.33, 176.0, 176.49, 174.87, 0.01] # open

def get_actuate_joints(x):
    if len(x) == 11:
        indexes = [9, 0, 4, 6, 2, 8]
        radian = x[indexes]
        radian[1:-1] += np.pi
    else:
        radian = x.copy()
    
    limit1_rad = np.array(upper_limits) * np.pi / 180
    limit2_rad = np.array(lower_limits) * np.pi / 180
    limit_min = np.minimum(limit1_rad, limit2_rad)
    limit_max = np.maximum(limit1_rad, limit2_rad)

    radian = np.clip(radian, limit_min, limit_max)
    return radian

class OYhandNode:
    def __init__(self):
        
        # robot hand control
        COM_PORT = '/dev/ttyUSB0'
        self.NODE_ID = 2
        
        self.client = ModbusSerialClient(
            port=COM_PORT,
            baudrate=115200,
            framer='rtu'
        )
        self.client.connect()
        
    def grasp(self):
        # thumb bend, index, middle, ring, little, thumb rot
        self.set_pos([0.3, 1.8, 1.8, 1.8, 1.8, 2])
        
    def open(self):
        self.set_pos([0.64, 3., 3., 3., 3., 0])
        
    def thumb_back(self):
        self.set_pos([0.64, 3.14, 3.14, 3.14, 3.14, 1.57])
    
    def yeah(self):
        self.set_pos([0.1, 3.14, 3.14, 2.0, 2.0, 3.14])
        
    def rock(self):
        self.set_pos([0.3, 3.14, 2.0, 2.0, 3.14, 3])
        
    

    def set_pos(self, pose):
        
        '''
        If pose is 11-dim:
            pose[0, 1] is the index finger joint
            pose[2, 3] is the little finger joint 
            pose[4, 5] is the middle finger joint
            pose[6, 7] is the ring finger joint
            pose[8] if is thumb rotation joins
            pose[9, 10] is the thumb bending joint
            
        If pose is 6-dim:
            pose[0] if the thumb bending
            pose[1] is the index finger joint
            pose[2] is the middle finger joint
            pose[3] is the ring finger joint
            pose[4] is the little finger joint
            pose[5] is the thumb rotation joint
        '''
        
        # Convert target position to a numpy array
        target_pos = np.array(pose)
        
        # Calculate the difference between target and current position
        targets = get_actuate_joints(target_pos)
        cmd = real_radian_to_cmd(targets)
        # print(cmd[:])
        
        # write
        cmd = [int(value) for value in cmd]
        resp = self.client.write_registers(ROH_FINGER_ANGLE_TARGET0, cmd, slave=self.NODE_ID) # all six finger joints
        time.sleep(1/30)

    #read position
    def read_pos(self):
        resp = self.client.read_holding_registers(ROH_FINGER_ANGLE0, count=6, slave=self.NODE_ID)
        angles = resp.registers
        angles = cmd_to_real_radian(angles)
        return angles
    
    #read velocity
    def read_vel(self):
        raise NotImplementedError
    
    #read current
    def read_cur(self):
        raise NotImplementedError
    
def single_real_angle_to_cmd(real_angle):
    target_angle = np.round(real_angle * 100)
    # print(target_angle)

    if (target_angle < 0) :
        target_angle += 65536

    return target_angle

def real_angle_to_cmd(real_angle):
    if isinstance(real_angle, list):
        return [single_real_angle_to_cmd(angle) for angle in real_angle]
    elif isinstance(real_angle, np.ndarray):
        return np.array([single_real_angle_to_cmd(angle) for angle in real_angle])
    else:
        return single_real_angle_to_cmd(real_angle)

def real_radian_to_cmd(real_radian):
    if isinstance(real_radian, list):
        real_angle = [angle * 180 / np.pi for angle in real_radian]
    else: # for both scalar and np.ndarray
        real_angle = real_radian * 180 / np.pi
    return real_angle_to_cmd(real_angle)

def single_cmd_to_real_angle(cmd):
    if (cmd > 32767) :
        cmd -= 65535
    real_angle = cmd / 100.0
    return real_angle

def cmd_to_real_angle(cmd):
    if isinstance(cmd, list):
        return [single_cmd_to_real_angle(angle) for angle in cmd]
    elif isinstance(cmd, np.ndarray):
        return np.array([single_cmd_to_real_angle(angle) for angle in cmd])
    else:
        return single_cmd_to_real_angle(cmd)

def cmd_to_real_radian(cmd):
    real_angle = cmd_to_real_angle(cmd)
    if isinstance(real_angle, list):
        real_radian = [angle * np.pi / 180 for angle in real_angle]
    else: # for both scalar and np.ndarray
        real_radian = real_angle * np.pi / 180
    return real_radian


# ModBus-RTU registers for ROH

MODBUS_PROTOCOL_VERSION_MAJOR = 1

ROH_PROTOCOL_VERSION      = (1000) # R
ROH_FW_VERSION            = (1001) # R
ROH_FW_REVISION           = (1002) # R
ROH_HW_VERSION            = (1003) # R
ROH_BOOT_VERSION          = (1004) # R
ROH_NODE_ID               = (1005) # R/W
ROH_SUB_EXCEPTION         = (1006) # R
ROH_BATTERY_VOLTAGE       = (1007) # R
ROH_SELF_TEST_LEVEL       = (1008) # R/W
ROH_BEEP_SWITCH           = (1009) # R/W
ROH_BEEP_PERIOD           = (1010) # W
ROH_BUTTON_PRESS_CNT      = (1011) # R/W
ROH_RECALIBRATE           = (1012) # W
ROH_START_INIT            = (1013) # W
ROH_RESET                 = (1014) # W
ROH_POWER_OFF             = (1015) # W
ROH_RESERVED0             = (1016) # R/W
ROH_RESERVED1             = (1017) # R/W
ROH_RESERVED2             = (1018) # R/W
ROH_RESERVED3             = (1019) # R/W
ROH_CALI_END0             = (1020) # R/W
ROH_CALI_END1             = (1021) # R/W
ROH_CALI_END2             = (1022) # R/W
ROH_CALI_END3             = (1023) # R/W
ROH_CALI_END4             = (1024) # R/W
ROH_CALI_END5             = (1025) # R/W
ROH_CALI_END6             = (1026) # R/W
ROH_CALI_END7             = (1027) # R/W
ROH_CALI_END8             = (1028) # R/W
ROH_CALI_END9             = (1029) # R/W
ROH_CALI_START0           = (1030) # R/W
ROH_CALI_START1           = (1031) # R/W
ROH_CALI_START2           = (1032) # R/W
ROH_CALI_START3           = (1033) # R/W
ROH_CALI_START4           = (1034) # R/W
ROH_CALI_START5           = (1035) # R/W
ROH_CALI_START6           = (1036) # R/W
ROH_CALI_START7           = (1037) # R/W
ROH_CALI_START8           = (1038) # R/W
ROH_CALI_START9           = (1039) # R/W
ROH_CALI_THUMB_POS0       = (1040) # R/W
ROH_CALI_THUMB_POS1       = (1041) # R/W
ROH_CALI_THUMB_POS2       = (1042) # R/W
ROH_CALI_THUMB_POS3       = (1043) # R/W
ROH_CALI_THUMB_POS4       = (1044) # R/W
ROH_FINGER_P0             = (1045) # R/W
ROH_FINGER_P1             = (1046) # R/W
ROH_FINGER_P2             = (1047) # R/W
ROH_FINGER_P3             = (1048) # R/W
ROH_FINGER_P4             = (1049) # R/W
ROH_FINGER_P5             = (1050) # R/W
ROH_FINGER_P6             = (1051) # R/W
ROH_FINGER_P7             = (1052) # R/W
ROH_FINGER_P8             = (1053) # R/W
ROH_FINGER_P9             = (1054) # R/W
ROH_FINGER_I0             = (1055) # R/W
ROH_FINGER_I1             = (1056) # R/W
ROH_FINGER_I2             = (1057) # R/W
ROH_FINGER_I3             = (1058) # R/W
ROH_FINGER_I4             = (1059) # R/W
ROH_FINGER_I5             = (1060) # R/W
ROH_FINGER_I6             = (1061) # R/W
ROH_FINGER_I7             = (1062) # R/W
ROH_FINGER_I8             = (1063) # R/W
ROH_FINGER_I9             = (1064) # R/W
ROH_FINGER_D0             = (1065) # R/W
ROH_FINGER_D1             = (1066) # R/W
ROH_FINGER_D2             = (1067) # R/W
ROH_FINGER_D3             = (1068) # R/W
ROH_FINGER_D4             = (1069) # R/W
ROH_FINGER_D5             = (1070) # R/W
ROH_FINGER_D6             = (1071) # R/W
ROH_FINGER_D7             = (1072) # R/W
ROH_FINGER_D8             = (1073) # R/W
ROH_FINGER_D9             = (1074) # R/W
ROH_FINGER_G0             = (1075) # R/W
ROH_FINGER_G1             = (1076) # R/W
ROH_FINGER_G2             = (1077) # R/W
ROH_FINGER_G3             = (1078) # R/W
ROH_FINGER_G4             = (1079) # R/W
ROH_FINGER_G5             = (1080) # R/W
ROH_FINGER_G6             = (1081) # R/W
ROH_FINGER_G7             = (1082) # R/W
ROH_FINGER_G8             = (1083) # R/W
ROH_FINGER_G9             = (1084) # R/W
ROH_FINGER_STATUS0        = (1085) # R
ROH_FINGER_STATUS1        = (1086) # R
ROH_FINGER_STATUS2        = (1087) # R
ROH_FINGER_STATUS3        = (1088) # R
ROH_FINGER_STATUS4        = (1089) # R
ROH_FINGER_STATUS5        = (1090) # R
ROH_FINGER_STATUS6        = (1091) # R
ROH_FINGER_STATUS7        = (1092) # R
ROH_FINGER_STATUS8        = (1093) # R
ROH_FINGER_STATUS9        = (1094) # R
ROH_FINGER_CURRENT_LIMIT0 = (1095) # R/W
ROH_FINGER_CURRENT_LIMIT1 = (1096) # R/W
ROH_FINGER_CURRENT_LIMIT2 = (1097) # R/W
ROH_FINGER_CURRENT_LIMIT3 = (1098) # R/W
ROH_FINGER_CURRENT_LIMIT4 = (1099) # R/W
ROH_FINGER_CURRENT_LIMIT5 = (1100) # R/W
ROH_FINGER_CURRENT_LIMIT6 = (1101) # R/W
ROH_FINGER_CURRENT_LIMIT7 = (1102) # R/W
ROH_FINGER_CURRENT_LIMIT8 = (1103) # R/W
ROH_FINGER_CURRENT_LIMIT9 = (1104) # R/W
ROH_FINGER_CURRENT0       = (1105) # R
ROH_FINGER_CURRENT1       = (1106) # R
ROH_FINGER_CURRENT2       = (1107) # R
ROH_FINGER_CURRENT3       = (1108) # R
ROH_FINGER_CURRENT4       = (1109) # R
ROH_FINGER_CURRENT5       = (1110) # R
ROH_FINGER_CURRENT6       = (1111) # R
ROH_FINGER_CURRENT7       = (1112) # R
ROH_FINGER_CURRENT8       = (1113) # R
ROH_FINGER_CURRENT9       = (1114) # R
ROH_FINGER_FORCE_LIMIT0   = (1115) # R/W
ROH_FINGER_FORCE_LIMIT1   = (1116) # R/W
ROH_FINGER_FORCE_LIMIT2   = (1117) # R/W
ROH_FINGER_FORCE_LIMIT3   = (1118) # R/W
ROH_FINGER_FORCE_LIMIT4   = (1119) # R/W
ROH_FINGER_FORCE0         = (1120) # R
ROH_FINGER_FORCE1         = (1121) # R
ROH_FINGER_FORCE2         = (1122) # R
ROH_FINGER_FORCE3         = (1123) # R
ROH_FINGER_FORCE4         = (1124) # R
ROH_FINGER_SPEED0         = (1125) # R/W
ROH_FINGER_SPEED1         = (1126) # R/W
ROH_FINGER_SPEED2         = (1127) # R/W
ROH_FINGER_SPEED3         = (1128) # R/W
ROH_FINGER_SPEED4         = (1129) # R/W
ROH_FINGER_SPEED5         = (1130) # R/W
ROH_FINGER_SPEED6         = (1131) # R/W
ROH_FINGER_SPEED7         = (1132) # R/W
ROH_FINGER_SPEED8         = (1133) # R/W
ROH_FINGER_SPEED9         = (1134) # R/W
ROH_FINGER_POS_TARGET0    = (1135) # R/W
ROH_FINGER_POS_TARGET1    = (1136) # R/W
ROH_FINGER_POS_TARGET2    = (1137) # R/W
ROH_FINGER_POS_TARGET3    = (1138) # R/W
ROH_FINGER_POS_TARGET4    = (1139) # R/W
ROH_FINGER_POS_TARGET5    = (1140) # R/W
ROH_FINGER_POS_TARGET6    = (1141) # R/W
ROH_FINGER_POS_TARGET7    = (1142) # R/W
ROH_FINGER_POS_TARGET8    = (1143) # R/W
ROH_FINGER_POS_TARGET9    = (1144) # R/W
ROH_FINGER_POS0           = (1145) # R
ROH_FINGER_POS1           = (1146) # R
ROH_FINGER_POS2           = (1147) # R
ROH_FINGER_POS3           = (1148) # R
ROH_FINGER_POS4           = (1149) # R
ROH_FINGER_POS5           = (1150) # R
ROH_FINGER_POS6           = (1151) # R
ROH_FINGER_POS7           = (1152) # R
ROH_FINGER_POS8           = (1153) # R
ROH_FINGER_POS9           = (1154) # R
ROH_FINGER_ANGLE_TARGET0  = (1155) # R/W
ROH_FINGER_ANGLE_TARGET1  = (1156) # R/W
ROH_FINGER_ANGLE_TARGET2  = (1157) # R/W
ROH_FINGER_ANGLE_TARGET3  = (1158) # R/W
ROH_FINGER_ANGLE_TARGET4  = (1159) # R/W
ROH_FINGER_ANGLE_TARGET5  = (1160) # R/W
ROH_FINGER_ANGLE_TARGET6  = (1161) # R/W
ROH_FINGER_ANGLE_TARGET7  = (1162) # R/W
ROH_FINGER_ANGLE_TARGET8  = (1163) # R/W
ROH_FINGER_ANGLE_TARGET9  = (1164) # R/W
ROH_FINGER_ANGLE0         = (1165) # R
ROH_FINGER_ANGLE1         = (1166) # R
ROH_FINGER_ANGLE2         = (1167) # R
ROH_FINGER_ANGLE3         = (1168) # R
ROH_FINGER_ANGLE4         = (1169) # R
ROH_FINGER_ANGLE5         = (1170) # R
ROH_FINGER_ANGLE6         = (1171) # R
ROH_FINGER_ANGLE7         = (1172) # R
ROH_FINGER_ANGLE8         = (1173) # R
ROH_FINGER_ANGLE9         = (1174) # R

if __name__ == '__main__':
    oyhand = OYhandNode()
    
    oyhand.yeah()
    time.sleep(5)
    
    oyhand.open()
    time.sleep(2)
    
    oyhand.rock()
    time.sleep(5)
    
    # oyhand.thumb_back()
    
    oyhand.open()
    time.sleep(2)
    
    oyhand.grasp()
    
    time.sleep(5)
    
    oyhand.open()