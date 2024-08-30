import numpy as np

chi2_channel_gain_fading = [0.0735074607347625, 0.02749778906619845, 0.029174260010851022, 0.009423351337083406,
                            0.007406596127356058, 0.035128271627365176, 0.0044334236068752815, 0.0011037399657171675,
                            0.08049370307647478, 0.023876009585841, 0.09676339809888745, 0.0588623929841733,
                            0.16881065185124564,
                            0.00822082327523068, 0.008083236325030742, 0.010635789611302949, 0.035443737632139434,
                            0.005465287765243086, 0.02733973869531951, 0.013390259328201974, 0.0475634329788755,
                            0.144090960693152,
                            0.04783696033090855, 0.014042502842292656, 0.03387971191046962, 0.054243997173709486,
                            0.13328998779945272, 0.02429668631926457, 0.019087604836640716, 0.0020223104059964056,
                            0.009596130153526285, 0.006487510943978741, 0.007792760059817744, 0.08023924107989569,
                            0.06063420648460928, 0.05170237108354504, 0.14960483181707637, 0.11775555555703365,
                            0.014118032487447325,
                            0.054781255679444044, 0.005414435362398907, 0.047398991947219034, 0.0732335697169703,
                            0.06927507455756406, 0.02329600669174674, 0.0337832711137338, 0.09900478824216602,
                            0.006539348837605203,
                            0.06130982954109258, 0.16738946145498704]
rayleigh_fading = [0.0065607923585634725, 0.0003086006040621854, 0.007821525010089422, 0.04499378860252923,
                   0.00043221568594442236, 0.024936906996908363, 0.02205211833653932, 0.009221994794935103,
                   0.04613786335182848, 0.034294869785955556, 0.07947894356873841, 0.025062982324445676,
                   0.06749863915123983, 0.011380054971952299, 0.035928363734958335, 0.15415860715457538,
                   0.09974829866184809, 0.035541178709458734, 0.03739404120555866, 0.041854700710917084,
                   0.0031441993395882844, 0.03583638825403363, 0.1140542112356012, 0.0016736456772355793,
                   0.011240769934082413, 0.011649718796647784, 0.09472342389133427, 0.020233898077687898,
                   0.003879676491363317, 0.014867316478651369, 0.03549203160648971, 0.13549018483348516,
                   0.010898236264212083, 0.022887994255403343, 0.032526051884692535, 0.1569960208171309,
                   0.16284066894431073, 0.07964097887923109, 0.02856235656259083, 0.06688934395815323,
                   0.08223198454262715, 0.17918961514114887, 0.03217916072287879, 0.0595223716327516,
                   0.03294838094677518, 0.020132231488336805, 0.03272108301383592, 0.001258864752218688,
                   0.0403806286614051, 0.12412113494716545]

user_positions = [
    [4, 19], [64, 12], [38, 14], [22, 45], [23, 33], [1, 46], [56, 22], [38, 19], [4, 3], [20, 57],
    [44, 18], [49, 27], [52, 49], [55, 50], [23, 27], [48, 40], [30, 7], [22, 31], [5, 55], [33, 25],
    [41, 3], [45, 61], [44, 41], [48, 57], [30, 56], [51, 29], [10, 20], [59, 63], [54, 67], [32, 19],
    [5, 18], [5, 21], [32, 55], [5, 24], [31, 26], [0, 33], [15, 20], [56, 53], [57, 22], [0, 60],
    [53, 8], [45, 4], [63, 45], [54, 52], [2, 19], [24, 31], [25, 18], [9, 12], [7, 4], [0, 14]
]
cluster = [
    1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1
]
uav_positions = [
    [49.86953366, 32.88788339],
    [14.11916185, 25.57281582]
]
power_allocation = [0.2442, 0.1496, 0.0023, 0.8085, 0.8843, 0.2360, 0.7458, 0.4322, 0.1424, 0.0595,
 0.3038, 0.6383, 0.9443, 0.5155, 0.1747, 0.8468, 0.1232, 0.6059, 0.2341, 0.7077,
 0.1448, 0.0309, 0.8843, 0.4037, 0.6073, 0.8505, 0.1879, 0.4916, 0.8536, 0.1198,
 0.8817, 0.3462, 0.7929, 0.9332, 0.4745, 0.2439, 0.8795, 0.7178, 0.4622, 0.9614,
 0.7766, 0.4756, 0.3943, 0.0753, 0.9124, 0.2906, 0.2855, 0.2420, 0.8456, 0.8572]

max_power=[]


class WirelessCommunication:
    def __init__(self):

        self.c = 3e8  # Speed of light in m/s
        self.transmitter_gain = 2  # Transmitter gain (linear scale)
        self.receiver_gain = 1.58  # Receiver gain (linear scale)
        self.frequency = 2.4e9  # Frequency in Hz 2.4Ghz
        self.noise = 1  # Noise power in watts
        self.bandwidth = 20e6  # Bandwidth in Hz

        for index_uav, uav in enumerate(uav_positions):
            m = 0
            for index_power, power in enumerate(power_allocation):
                if cluster[index_power] == index_uav:
                    m = max(power, m)
            max_power.append(m)


    def calculate_served_user_number(self):
        served_user_number=[]
        for index_uav,uav in enumerate(uav_positions) :
            number=0
            for index_user,user in enumerate(user_positions):
                if cluster[index_user]==index_uav:
                    user_position=user_positions[index_user]
                    uav_position=uav_positions[index_uav]
                    distance = self.calculate_distance(user_position, uav_position)
                    received_power = self.calculate_received_power(distance, power_allocation[index_user])
                    interference=self.calculate_interference(index_uav,index_user)
                    sinr= self.calculate_sinr(received_power,interference)
                    data_rate=self.calculate_channel_capacity(index_user,sinr,distance)
                    if data_rate>1:
                        number+=1
            served_user_number.append(number)
        return served_user_number


    def calculate_distance(self, position1, position2):
        return np.linalg.norm(position1 - position2)

    def calculate_received_power(self,distance,power):
        path_loss = (self.c / (4 * np.pi * distance * self.frequency)) ** 2
        return power * path_loss * self.transmitter_gain * self.receiver_gain

    def calculate_sinr(self,received_power,interference):
        return received_power / (self.noise + interference)

    def calculate_interference(self,index_uav,index_user):
        interference=0
        for index,uav in enumerate(uav_positions):
            if index!=index_uav:
                distance = self.calculate_distance(user_positions[index_user], uav_positions[index])
                i=self.calculate_received_power(distance, max_power[index])
                interference+=i
        return interference


    def calculate_channel_capacity(self,index, sinr, distance):
        c = 11.95
        b = 0.136
        sinr_line_of_sight = chi2_channel_gain_fading[index]*sinr
        sinr_none_line_of_sight = rayleigh_fading[index]*sinr

        # theta degree
        theta = (180 / np.pi) * np.arcsin(self.drones[self.cluster].height / distance)

        # probability
        probability_line_of_sight = 1 / (1 + c * np.exp(-b * (theta - c)))
        probability_none_line_of_sight = 1 - probability_line_of_sight

        return self.bandwidth * np.log2(1 + sinr_line_of_sight)*probability_line_of_sight+self.bandwidth * np.log2(1 + sinr_none_line_of_sight)*probability_none_line_of_sight


new = WirelessCommunication()
print(new.calculate_served_user_number())