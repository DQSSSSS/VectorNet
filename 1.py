# # from argoverse.map_representation.map_api import ArgoverseMap
# import pandas as pd
# import numpy as np
#
# avm = ArgoverseMap()
# a = avm.build_centerline_index()
#
# X_ID = 3
# Y_ID = 4
#
#
# def vecLink(a, polyID, AVTIME):
#     a = np.array(a)
#     ans = []
#     type = 0 if a[0, 2] == 'AGENT' else 1
#     for i in range(a.shape[0] - 1):
#         l, r = a[i], a[i + 1]
#         if type == 1 and (l[0] > AVTIME or r[0] > AVTIME):
#             break
#         now = [l[X_ID], l[Y_ID], r[X_ID], r[Y_ID], type,
#                l[0],
#                r[0],
#                np.sqrt(np.square(l[X_ID] - r[X_ID]) + np.square(l[Y_ID] - r[Y_ID])) / (r[0] - l[0]),
#                polyID]
#         ans.append(now)
#     return ans
#
#
# def work(name, file, ooo):
#     ans = pd.read_csv(name)
#     ans = np.array(ans)
#
#     city = ans[0][-1]
#     track_id = 1
#
#     id = np.argsort(ans[:, 0], kind='mergesort')
#     tmp = np.zeros_like(ans)
#     for i in range(ans.shape[0]):
#         tmp[i] = ans[id[i]]
#     ans = tmp
#
#     id = np.argsort(ans[:, 1], kind='mergesort')
#     tmp = np.zeros_like(ans)
#     for i in range(ans.shape[0]):
#         tmp[i] = ans[id[i]]
#     ans = tmp
#
#     # print(ans)
#
#     AVX = 0
#     AVY = 0
#     AVTIME = 0
#
#     for i in range(ans.shape[0]):
#         if i + 1 == ans.shape[0] or \
#                 ans[i, track_id] != ans[i + 1, track_id]:
#             if ans[i, 2] == 'AGENT':
#                 AVX, AVY = ans[i - 30, 3], ans[i - 30, 4]
#                 AVTIME = ans[i - 30, 0]
#     tmp = []
#     j = 0
#     polyID = 0
#     for i in range(ans.shape[0]):
#         if i + 1 == ans.shape[0] or \
#                 ans[i, track_id] != ans[i + 1, track_id]:
#             now = []
#             while j <= i:
#                 now.append(ans[j])
#                 if j < i:
#                     assert ans[j, 0] <= ans[j + 1, 0]
#                 j += 1
#             vecList = vecLink(now, polyID, AVTIME)
#             polyID += 1
#             for vec in vecList:
#                 tmp.append(vec)
#
#     idList = avm.get_lane_ids_in_xy_bbox(AVX, AVY, city, 65)
#
#     for id in idList:
#         lane = a[city][id]
#         #        print(lane.id)
#         #        print(lane.has_traffic_control)
#         #        print(lane.turn_direction)
#         #        print(lane.is_intersection)
#         #        print(lane.centerline)
#
#         ans = []
#         for i in range(lane.centerline.shape[0] - 1):
#             l, r = lane.centerline[i], lane.centerline[i + 1]
#
#             t = 0
#             if lane.turn_direction == 'LEFT':
#                 t = 1
#             elif lane.turn_direction == 'RIGHT':
#                 t = 2
#
#             now = [l[0], l[1], r[0], r[1], 2,
#                    0 if lane.has_traffic_control == False else 1,
#                    t,
#                    0 if lane.is_intersection == False else 1,
#                    polyID]
#
#             tmp.append(now)
#         polyID += 1
#
#     tmp = np.array(tmp)
#     for i in range(tmp.shape[0]):
#         tmp[i, 0] -= AVX
#         tmp[i, 2] -= AVX
#         tmp[i, 1] -= AVY
#         tmp[i, 3] -= AVY
#         # for j in range(4):
#         #    tmp[i , j] *= 100
#         if tmp[i, 4] != 2:
#             tmp[i, 5] -= AVTIME
#             tmp[i, 6] -= AVTIME
#
#     # print(tmp)
#     print(tmp.shape)
#     pf = pd.DataFrame(data=tmp)
#     pf.to_csv(ooo + 'data_' + file, header=False, index=False)
#
#
# if __name__ == '__main__':
#     path = 'argoverse-forecasting/data/forecasting_sample/data/'
#     nameList = ['2645.csv', '3700.csv', '3828.csv', '3861.csv', '4791.csv']
#     for name in nameList:
#         work(path + name, name, '')
#     path = 'data-f/val-data/'
#     nameList = ['10905.csv', '11523.csv', '12688.csv', '12945.csv', '15072.csv', '16049.csv', '16996.csv', '17471.csv',
#                 '18137.csv', '20170.csv', '20577.csv', '2468.csv', '26309.csv', '27477.csv', '27995.csv', '28378.csv',
#                 '28515.csv', '28681.csv', '2883.csv', '29398.csv', '29795.csv', '30090.csv', '30126.csv', '30389.csv',
#                 '30657.csv', '31444.csv', '31765.csv', '31951.csv', '32106.csv', '33032.csv', '33333.csv', '33566.csv',
#                 '3485.csv', '35058.csv', '35153.csv', '36203.csv', '36511.csv', '36678.csv', '37674.csv', '37751.csv',
#                 '37960.csv', '38183.csv', '38346.csv', '39114.csv', '39545.csv', '39604.csv', '40003.csv', '40128.csv',
#                 '4872.csv', '6473.csv', '7501.csv', '8007.csv', '9913.csv']
#     for name in nameList:
#         work(path + name, name, 'test-data/')
#
#     path = 'data-f/data-train/'
#     nameList = ['100015.csv', '101451.csv', '102423.csv', '103369.csv', '10503.csv', '105292.csv', '105327.csv',
#                 '105529.csv', '108080.csv', '108557.csv', '109076.csv', '109874.csv', '110673.csv', '111415.csv',
#                 '111870.csv', '113003.csv', '113137.csv', '113555.csv', '115988.csv', '116257.csv', '117092.csv',
#                 '117295.csv', '117323.csv', '11800.csv', '118563.csv', '123099.csv', '126725.csv', '127736.csv',
#                 '129602.csv', '130908.csv', '134225.csv', '136495.csv', '136839.csv', '139049.csv', '141651.csv',
#                 '146491.csv', '148506.csv', '148845.csv', '150891.csv', '155122.csv', '155444.csv', '155622.csv',
#                 '156762.csv', '157834.csv', '159493.csv', '159522.csv', '161459.csv', '163620.csv', '172532.csv',
#                 '173988.csv', '174662.csv', '176035.csv', '176664.csv', '178553.csv', '178929.csv', '179455.csv',
#                 '17993.csv', '18101.csv', '181246.csv', '187368.csv', '189291.csv', '190905.csv', '195493.csv',
#                 '19796.csv', '198714.csv', '201698.csv', '202667.csv', '2045.csv', '204886.csv', '205516.csv',
#                 '210073.csv', '21666.csv', '24051.csv', '2790.csv', '28682.csv', '31341.csv', '34674.csv', '35153.csv',
#                 '35546.csv', '37821.csv', '37921.csv', '39695.csv', '40043.csv', '40196.csv', '47123.csv', '497.csv',
#                 '50981.csv', '51052.csv', '51259.csv', '55503.csv', '56324.csv', '57386.csv', '6033.csv', '61312.csv',
#                 '62470.csv', '63764.csv', '64668.csv', '65255.csv', '66304.csv', '66892.csv', '68845.csv', '69218.csv',
#                 '73846.csv', '74312.csv', '77260.csv', '79390.csv', '79659.csv', '79815.csv', '79914.csv', '80340.csv',
#                 '82154.csv', '83076.csv', '84282.csv', '85064.csv', '85095.csv', '85876.csv', '86632.csv', '86719.csv',
#                 '87741.csv', '88506.csv', '9124.csv', '93097.csv', '93200.csv', '93686.csv', '95097.csv', '96589.csv',
#                 '96640.csv', '9787.csv']
#     for name in nameList:
#         work(path + name, name, 'train-data/')
#
#
