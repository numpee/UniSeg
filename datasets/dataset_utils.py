import numpy as np

label2train = {
    'cityscapes': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 0], [8, 1],
                   [9, 255], [10, 255],
                   [11, 2], [12, 3], [13, 4], [14, 255], [15, 255], [16, 255], [17, 5], [18, 255], [19, 6],
                   [20, 7],
                   [21, 8], [22, 9], [23, 10], [24, 11], [25, 12], [26, 13], [27, 14], [28, 15], [29, 255],
                   [30, 255],
                   [31, 16], [32, 17], [33, 18], [-1, 255]],
    'idd': [[0, 0], [1, 1], [2, 1], [3, 2], [4, 3], [5, 3], [6, 4], [7, 4], [8, 5], [9, 6], [10, 7], [11, 8],
            [12, 9], [13, 10],
            [14, 11], [15, 12], [16, 12], [17, 12], [18, 12], [19, 13], [20, 14], [21, 15], [22, 16],
            [23, 17],
            [24, 18], [25, 19], [26, 20], [27, 20], [28, 21], [29, 22], [30, 23], [31, 23], [32, 24],
            [33, 25],

            [34, 25], [35, 255], [36, 255], [37, 255], [38, 255], [39, 255]],
    'bdd': [[i, i] for i in range(19)] + [[255, 255]],
    'kitti': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 0], [8, 1], [9, 255], [10, 255],
              [11, 2], [12, 3], [13, 4], [14, 255], [15, 255], [16, 255], [17, 5], [18, 255], [19, 6], [20, 7], [21, 8],
              [22, 9], [23, 10], [24, 11], [25, 12], [26, 13], [27, 14], [28, 15], [29, 255], [30, 255], [31, 16],
              [32, 17], [33, 18], [-1, 255]],
    'wilddash': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 0], [8, 1], [9, 255],
                 [10, 255], [11, 2], [12, 3], [13, 4], [14, 255], [15, 255], [16, 255], [17, 5], [18, 255], [19, 6],
                 [20, 7], [21, 8], [22, 9], [23, 10], [24, 11], [25, 12], [26, 13], [27, 14], [28, 15], [29, 255],
                 [30, 255], [31, 16], [32, 17], [33, 18], [-1, 255]],
    'mapillary': [[i, i] for i in range(65)] + [[65, 255]],
    'camvid': [[i, i] for i in range(11)] + [[-1, 255]],
    'camvid_v2': [[i, i] for i in range(12)] + [[-1, 255]],
    'nuscenes': [[i, i] for i in range(14)],

    # Mappings for CIB
    'idd_to_cib': [[0, 17], [1, 12], [2, 12], [3, 18], [4, 15], [5, 15], [6, 13], [7, 13], [8, 16], [9, 10], [10, 1],
                   [11, 0], [12, 6], [13, 24], [14, 5], [15, 26], [16, 26], [17, 26], [18, 26], [19, 7], [20, 27],
                   [21, 8], [22, 9], [23, 2], [24, 22], [25, 21], [26, 14], [27, 14], [28, 11], [29, 4], [30, 3],
                   [31, 3], [32, 25], [33, 19], [34, 19], [35, 255], [36, 255], [37, 255], [38, 255], [39, 255]],
    'cityscapes_to_cib': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 17], [8, 18],
                          [9, 255], [10, 255], [11, 4], [12, 27], [13, 8], [14, 255], [15, 255], [16, 255], [17, 14],
                          [18, 255], [19, 21], [20, 22], [21, 25], [22, 20], [23, 19], [24, 13], [25, 16], [26, 6],
                          [27, 24], [28, 5], [29, 255], [30, 255], [31, 23], [32, 10], [33, 1], [-1, 255]],
    'bdd_to_cib': [[0, 17], [1, 18], [2, 4], [3, 27], [4, 8], [5, 14], [6, 21], [7, 22], [8, 25], [9, 20], [10, 19],
                   [11, 13], [12, 16], [13, 6], [14, 24], [15, 5], [16, 23], [17, 10], [18, 1], [255, 255]],
    'wilddash_to_cib': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 17], [8, 18],
                        [9, 255], [10, 255], [11, 4], [12, 27], [13, 8], [14, 255], [15, 255], [16, 255], [17, 14],
                        [18, 255], [19, 21], [20, 22], [21, 25], [22, 20], [23, 19], [24, 13], [25, 16], [26, 6],
                        [27, 24], [28, 5], [29, 255], [30, 255], [31, 23], [32, 10], [33, 1], [-1, 255]],
    'kitti_to_cib': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 17], [8, 18], [9, 255],
                     [10, 255], [11, 4], [12, 27], [13, 8], [14, 255], [15, 255], [16, 255], [17, 14], [18, 255],
                     [19, 21], [20, 22], [21, 25], [22, 20], [23, 19], [24, 13], [25, 16], [26, 6], [27, 24], [28, 5],
                     [29, 255], [30, 255], [31, 23], [32, 10], [33, 1], [-1, 255]],

    # Mappings for IBM
    'idd_to_ibm': [[0, 47], [1, 39], [2, 39], [3, 50], [4, 45], [5, 45], [6, 41], [7, 41], [8, 46], [9, 32], [10, 4],
                   [11, 0], [12, 14], [13, 62], [14, 13], [15, 66], [16, 66], [17, 66], [18, 66], [19, 20], [20, 67],
                   [21, 23], [22, 26], [23, 8], [24, 56], [25, 55], [26, 43], [27, 43], [28, 35], [29, 12], [30, 11],
                   [31, 11], [32, 65], [33, 51], [34, 51], [35, 255], [36, 255], [37, 255], [38, 255], [39, 255]],
    'mapillary_to_ibm': [[0, 9], [1, 25], [2, 20], [3, 23], [4, 26], [5, 2], [6, 67], [7, 6], [8, 19], [9, 21],
                         [10, 39], [11, 40], [12, 45], [13, 47], [14, 49], [15, 50], [16, 11], [17, 12], [18, 63],
                         [19, 41], [20, 5], [21, 33], [22, 37], [23, 28], [24, 29], [25, 34], [26, 48], [27, 51],
                         [28, 52], [29, 54], [30, 65], [31, 68], [32, 1], [33, 3], [34, 7], [35, 8], [36, 17], [37, 18],
                         [38, 24], [39, 27], [40, 30], [41, 31], [42, 42], [43, 44], [44, 53], [45, 43], [46, 58],
                         [47, 64], [48, 55], [49, 57], [50, 56], [51, 61], [52, 4], [53, 10], [54, 13], [55, 14],
                         [56, 16], [57, 32], [58, 36], [59, 38], [60, 59], [61, 62], [62, 69], [63, 15], [64, 22],
                         [65, 255]],
    'bdd_to_ibm': [[0, 47], [1, 50], [2, 12], [3, 67], [4, 23], [5, 43], [6, 55], [7, 56], [8, 65], [9, 54], [10, 51],
                   [11, 41], [12, 46], [13, 14], [14, 62], [15, 13], [16, 60], [17, 32], [18, 4], [255, 255]],
    'wilddash_to_ibm': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 47], [8, 50],
                        [9, 255], [10, 255], [11, 12], [12, 67], [13, 23], [14, 255], [15, 255], [16, 255], [17, 43],
                        [18, 255], [19, 55], [20, 56], [21, 65], [22, 54], [23, 51], [24, 41], [25, 46], [26, 14],
                        [27, 62], [28, 13], [29, 255], [30, 255], [31, 60], [32, 32], [33, 4], [-1, 255]],
    'kitti_to_ibm': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 47], [8, 50], [9, 255],
                     [10, 255], [11, 12], [12, 67], [13, 23], [14, 255], [15, 255], [16, 255], [17, 43], [18, 255],
                     [19, 55], [20, 56], [21, 65], [22, 54], [23, 51], [24, 41], [25, 46], [26, 14], [27, 62], [28, 13],
                     [29, 255], [30, 255], [31, 60], [32, 32], [33, 4], [-1, 255]],
    # Mappings for BMC
    'mapillary_to_bmc': [[0, 8], [1, 24], [2, 19], [3, 22], [4, 25], [5, 1], [6, 64], [7, 5], [8, 18], [9, 20],
                         [10, 37], [11, 38], [12, 43], [13, 45], [14, 47], [15, 48], [16, 10], [17, 11], [18, 61],
                         [19, 39], [20, 4], [21, 32], [22, 35], [23, 27], [24, 28], [25, 33], [26, 46], [27, 49],
                         [28, 50], [29, 52], [30, 63], [31, 65], [32, 0], [33, 2], [34, 6], [35, 7], [36, 16], [37, 17],
                         [38, 23], [39, 26], [40, 29], [41, 30], [42, 40], [43, 42], [44, 51], [45, 41], [46, 56],
                         [47, 62], [48, 53], [49, 55], [50, 54], [51, 59], [52, 3], [53, 9], [54, 12], [55, 13],
                         [56, 15], [57, 31], [58, 34], [59, 36], [60, 57], [61, 60], [62, 66], [63, 14], [64, 21],
                         [65, 255]],
    'cityscapes_to_bmc': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 45], [8, 48],
                          [9, 255], [10, 255], [11, 11], [12, 64], [13, 22], [14, 255], [15, 255], [16, 255], [17, 41],
                          [18, 255], [19, 53], [20, 54], [21, 63], [22, 52], [23, 49], [24, 39], [25, 44], [26, 13],
                          [27, 60], [28, 12], [29, 255], [30, 255], [31, 58], [32, 31], [33, 3], [-1, 255]],
    'bdd_to_bmc': [[0, 45], [1, 48], [2, 11], [3, 64], [4, 22], [5, 41], [6, 53], [7, 54], [8, 63], [9, 52], [10, 49],
                   [11, 39], [12, 44], [13, 13], [14, 60], [15, 12], [16, 58], [17, 31], [18, 3], [255, 255]],

    # Mappings for CIM
    'mapillary_to_cim': [[0, 9], [1, 25], [2, 20], [3, 23], [4, 26], [5, 2], [6, 67], [7, 6], [8, 19], [9, 21],
                         [10, 39], [11, 40], [12, 45], [13, 47], [14, 49], [15, 50], [16, 11], [17, 12], [18, 63],
                         [19, 41], [20, 5], [21, 33], [22, 37], [23, 28], [24, 29], [25, 34], [26, 48], [27, 51],
                         [28, 52], [29, 54], [30, 65], [31, 68], [32, 1], [33, 3], [34, 7], [35, 8], [36, 17], [37, 18],
                         [38, 24], [39, 27], [40, 30], [41, 31], [42, 42], [43, 44], [44, 53], [45, 43], [46, 58],
                         [47, 64], [48, 55], [49, 57], [50, 56], [51, 61], [52, 4], [53, 10], [54, 13], [55, 14],
                         [56, 16], [57, 32], [58, 36], [59, 38], [60, 59], [61, 62], [62, 69], [63, 15], [64, 22],
                         [65, 255]],
    'idd_to_cim': [[0, 47], [1, 39], [2, 39], [3, 50], [4, 45], [5, 45], [6, 41], [7, 41], [8, 46], [9, 32], [10, 4],
                   [11, 0], [12, 14], [13, 62], [14, 13], [15, 66], [16, 66], [17, 66], [18, 66], [19, 20], [20, 67],
                   [21, 23], [22, 26], [23, 8], [24, 56], [25, 55], [26, 43], [27, 43], [28, 35], [29, 12], [30, 11],
                   [31, 11], [32, 65], [33, 51], [34, 51], [35, 255], [36, 255], [37, 255], [38, 255], [39, 255]],
    'cityscapes_to_cim': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 47], [8, 50],
                          [9, 255], [10, 255], [11, 12], [12, 67], [13, 23], [14, 255], [15, 255], [16, 255], [17, 43],
                          [18, 255], [19, 55], [20, 56], [21, 65], [22, 54], [23, 51], [24, 41], [25, 46], [26, 14],
                          [27, 62], [28, 13], [29, 255], [30, 255], [31, 60], [32, 32], [33, 4], [-1, 255]],

    # Mappings for City, IDD, BDD, Mapillary
    'cityscapes_to_combined': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 47], [8, 50],
                               [9, 255], [10, 255], [11, 12], [12, 67], [13, 23], [14, 255], [15, 255], [16, 255],
                               [17, 43], [18, 255], [19, 55], [20, 56], [21, 65], [22, 54], [23, 51], [24, 41],
                               [25, 46], [26, 14], [27, 62], [28, 13], [29, 255], [30, 255], [31, 60], [32, 32],
                               [33, 4], [-1, 255]],
    'bdd_to_combined': [[0, 47], [1, 50], [2, 12], [3, 67], [4, 23], [5, 43], [6, 55], [7, 56], [8, 65], [9, 54],
                        [10, 51], [11, 41], [12, 46], [13, 14], [14, 62], [15, 13], [16, 60], [17, 32], [18, 4],
                        [255, 255]],
    'idd_to_combined': [[0, 47], [1, 39], [2, 39], [3, 50], [4, 45], [5, 45], [6, 41], [7, 41], [8, 46], [9, 32],
                        [10, 4],
                        [11, 0], [12, 14], [13, 62], [14, 13], [15, 66], [16, 66], [17, 66], [18, 66], [19, 20],
                        [20, 67], [21, 23], [22, 26], [23, 8], [24, 56], [25, 55], [26, 43], [27, 43], [28, 35],
                        [29, 12], [30, 11], [31, 11], [32, 65], [33, 51], [34, 51], [35, 255], [36, 255], [37, 255],
                        [38, 255], [39, 255]],
    'mapillary_to_combined': [[0, 9], [1, 25], [2, 20], [3, 23], [4, 26], [5, 2], [6, 67], [7, 6], [8, 19], [9, 21],
                              [10, 39], [11, 40], [12, 45], [13, 47], [14, 49], [15, 50], [16, 11], [17, 12], [18, 63],
                              [19, 41], [20, 5], [21, 33], [22, 37], [23, 28], [24, 29], [25, 34], [26, 48], [27, 51],
                              [28, 52], [29, 54], [30, 65], [31, 68], [32, 1], [33, 3], [34, 7], [35, 8], [36, 17],
                              [37, 18], [38, 24], [39, 27], [40, 30], [41, 31], [42, 42], [43, 44], [44, 53], [45, 43],
                              [46, 58], [47, 64], [48, 55], [49, 57], [50, 56], [51, 61], [52, 4], [53, 10], [54, 13],
                              [55, 14], [56, 16], [57, 32], [58, 36], [59, 38], [60, 59], [61, 62], [62, 69], [63, 15],
                              [64, 22], [65, 255]],
    'kitti_to_combined': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 47], [8, 50],
                          [9, 255], [10, 255], [11, 12], [12, 67], [13, 23], [14, 255], [15, 255], [16, 255],
                          [17, 43], [18, 255], [19, 55], [20, 56], [21, 65], [22, 54], [23, 51], [24, 41],
                          [25, 46], [26, 14], [27, 62], [28, 13], [29, 255], [30, 255], [31, 60], [32, 32],
                          [33, 4], [-1, 255]],
    'wilddash_to_combined': [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 47], [8, 50],
                             [9, 255], [10, 255], [11, 12], [12, 67], [13, 23], [14, 255], [15, 255], [16, 255],
                             [17, 43], [18, 255], [19, 55], [20, 56], [21, 65], [22, 54], [23, 51], [24, 41],
                             [25, 46], [26, 14], [27, 62], [28, 13], [29, 255], [30, 255], [31, 60], [32, 32],
                             [33, 4], [-1, 255]],
    'camvid_to_combined': [[0, 5], [1, 12], [2, 14], [3, 23], [4, 41], [5, 43], [6, 47], [7, 50], [8, 51], [9, 56],
                           [10, 65], [-1, 255]],
    'camvid_v2_to_combined': [[0, 5], [1, 12], [2, 14], [3, 23], [4, 29], [5, 41], [6, 43], [7, 47], [8, 50], [9, 51],
                              [10, 56], [11, 65], [-1, 255]],
    # 'nuscenes_to_combined': [[0, 51], [1, 12], [2, 65], [3, 56], [4, 55], [5, 41], [6, 32], [7, 14], [8, 47], [9, 39],
    #                          [10, 45], [11, 50], [12, 19], [13, 29]],

    # mappings for training individual.yaml classifier (based on the combined mappings)
    # This is just the mapping from pixels --> classes in alphabetical order, so it is invariant to which combined
    # mapping set we use
    'cityscapes_to_comb_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                               [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                               [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                               [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'bdd_to_comb_ind': [[18, 0], [2, 1], [15, 2], [13, 3], [4, 4], [17, 5], [11, 6], [5, 7], [12, 8], [0, 9], [1, 10],
                        [10, 11], [9, 12], [6, 13], [7, 14], [16, 15], [14, 16], [8, 17], [3, 18], [255, 255]],
    'idd_to_comb_ind': [[11, 0], [10, 1], [23, 2], [30, 3], [31, 3], [29, 4], [14, 5], [12, 6], [19, 7], [21, 8],
                        [22, 9], [9, 10], [28, 11], [1, 12], [2, 12], [7, 13], [6, 13], [27, 14], [26, 14], [5, 15],
                        [4, 15], [8, 16], [0, 17], [3, 18], [33, 19], [34, 19], [25, 20], [24, 21], [13, 22], [32, 23],
                        [18, 24], [16, 24], [15, 24], [17, 24], [20, 25], [35, 255], [36, 255], [37, 255], [38, 255],
                        [39, 255]],
    'kitti_to_comb_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                          [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                          [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                          [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'wilddash_to_comb_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                             [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                             [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                             [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],

    'mapillary_to_comb_ind': [[0, 8], [1, 24], [2, 19], [3, 22], [4, 25], [5, 1], [6, 62], [7, 5], [8, 18], [9, 20],
                              [10, 37], [11, 38], [12, 43], [13, 44], [14, 46], [15, 47], [16, 10], [17, 11], [18, 59],
                              [19, 39], [20, 4], [21, 32], [22, 35], [23, 27], [24, 28], [25, 33], [26, 45], [27, 48],
                              [28, 49], [29, 51], [30, 61], [31, 63], [32, 0], [33, 2], [34, 6], [35, 7], [36, 16],
                              [37, 17], [38, 23], [39, 26], [40, 29], [41, 30], [42, 40], [43, 42], [44, 50], [45, 41],
                              [46, 55], [47, 60], [48, 52], [49, 54], [50, 53], [51, 57], [52, 3], [53, 9], [54, 12],
                              [55, 13], [56, 15], [57, 31], [58, 34], [59, 36], [60, 56], [61, 58], [62, 64], [63, 14],
                              [64, 21]],
    'camvid_to_comb_ind': [[i, i] for i in range(11)] + [[-1, 255]],
    'camvid_v2_to_comb_ind': [[i, i] for i in range(12)] + [[-1, 255]],

    ######## Mapping for CIB ########
    'mapillary_to_cib_ind': [[0, 255], [1, 255], [2, 6], [3, 7], [4, 8], [5, 255], [6, 22], [7, 255], [8, 255],
                             [9, 255], [10, 10], [11, 255], [12, 13], [13, 14], [14, 255], [15, 15], [16, 2], [17, 3],
                             [18, 255], [19, 11], [20, 255], [21, 255], [22, 255], [23, 255], [24, 255], [25, 255],
                             [26, 255], [27, 16], [28, 255], [29, 17], [30, 21], [31, 255], [32, 255], [33, 255],
                             [34, 255], [35, 1], [36, 255], [37, 255], [38, 255], [39, 255], [40, 255], [41, 255],
                             [42, 255], [43, 255], [44, 255], [45, 12], [46, 255], [47, 255], [48, 18], [49, 255],
                             [50, 19], [51, 255], [52, 0], [53, 255], [54, 4], [55, 5], [56, 255], [57, 9], [58, 255],
                             [59, 255], [60, 255], [61, 20], [62, 255], [63, 255], [64, 255], [65, 255]],
    'kitti_to_cib_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                         [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                         [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                         [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'wilddash_to_cib_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                            [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                            [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                            [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'camvid_to_cib_ind': [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [0, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10],
                          [-1, 255]],
    'mapillary_to_cib_ind_special': [[0, 255], [1, 255], [2, 6], [3, 7], [4, 8], [5, 255], [6, 22], [7, 14], [8, 14],
                                     [9, 255], [10, 10], [11, 255], [12, 13], [13, 14], [14, 14], [15, 15], [16, 2],
                                     [17, 3], [18, 255], [19, 11], [20, 255], [21, 255], [22, 255], [23, 14], [24, 14],
                                     [25, 255], [26, 255], [27, 16], [28, 255], [29, 17], [30, 21], [31, 255],
                                     [32, 255], [33, 255], [34, 255], [35, 1], [36, 255], [37, 255], [38, 255],
                                     [39, 255], [40, 255], [41, 255], [42, 255], [43, 255], [44, 255], [45, 12],
                                     [46, 255], [47, 255], [48, 18], [49, 19], [50, 19], [51, 255], [52, 0], [53, 255],
                                     [54, 4], [55, 5], [56, 255], [57, 9], [58, 255], [59, 255], [60, 255], [61, 20],
                                     [62, 255], [63, 255], [64, 255], [65, 255]],

    # Mapping for IBM (same as cityscapes_comb_ind)
    'cityscapes_to_ibm_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                              [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                              [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                              [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'kitti_to_ibm_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                         [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                         [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                         [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'wilddash_to_ibm_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                            [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                            [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                            [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'camvid_to_ibm_ind': [[i, i] for i in range(11)] + [[-1, 255]],

    # Mapping for BMC
    'idd_to_bmc_ind': [[0, 15], [1, 10], [2, 10], [3, 16], [4, 13], [5, 13], [6, 11], [7, 11], [8, 14], [9, 9], [10, 0],
                       [11, 255], [12, 5], [13, 20], [14, 4], [15, 255], [16, 255], [17, 255], [18, 255], [19, 6],
                       [20, 22], [21, 7], [22, 8], [23, 1], [24, 19], [25, 18], [26, 12], [27, 12], [28, 255], [29, 3],
                       [30, 2], [31, 2], [32, 21], [33, 17], [34, 17], [35, 255], [36, 255], [37, 255], [38, 255],
                       [39, 255]],
    'kitti_to_bmc_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                         [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                         [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                         [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'wilddash_to_bmc_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                            [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                            [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                            [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'camvid_to_bmc_ind': [[i, i] for i in range(11)] + [[-1, 255]],

    # Mapping for CIM
    'bdd_to_cim_ind': [[0, 9], [1, 10], [2, 1], [3, 18], [4, 4], [5, 7], [6, 13], [7, 14], [8, 17], [9, 12], [10, 11],
                       [11, 6], [12, 8], [13, 3], [14, 16], [15, 2], [16, 15], [17, 5], [18, 0], [255, 255]],
    'kitti_to_cim_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                         [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                         [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                         [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'wilddash_to_cim_ind': [[33, 0], [11, 1], [28, 2], [26, 3], [13, 4], [32, 5], [24, 6], [17, 7], [25, 8], [7, 9],
                            [8, 10], [23, 11], [22, 12], [19, 13], [20, 14], [31, 15], [27, 16], [21, 17], [12, 18],
                            [29, 255], [30, 255], [0, 255], [16, 255], [15, 255], [14, 255], [10, 255], [9, 255],
                            [6, 255], [5, 255], [4, 255], [3, 255], [2, 255], [1, 255], [18, 255], [-1, 255]],
    'camvid_to_cim_ind': [[i, i] for i in range(11)] + [[-1, 255]],

}

dataset_categories = {
    "cityscapes": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                   "bicycle"],
    "idd": ['road', 'parking', 'sidewalk', 'rail track', 'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw',
            'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard',
            'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky'],
    "bdd": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', "bicycle"],
    "kitti": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
              'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
              "bicycle"],
    "wilddash": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                 "bicycle"],
    "mapillary": ['bird', 'ground animal', 'curb', 'fence', 'guard rail', 'barrier', 'wall', 'bike lane',
                  'crosswalk - plain', 'curb cut', 'parking', 'pedestrian area', 'rail track', 'road', 'service lane',
                  'sidewalk', 'bridge', 'building', 'tunnel', 'person', 'bicyclist', 'motorcyclist', 'other rider',
                  'lane marking - crosswalk', 'lane marking - general', 'mountain', 'sand', 'sky', 'snow', 'terrain',
                  'vegetation', 'water', 'banner', 'bench', 'bike rack', 'billboard', 'catch basin', 'cctv camera',
                  'fire hydrant', 'junction box', 'mailbox', 'manhole', 'phone booth', 'pothole', 'street light',
                  'pole', 'traffic sign frame', 'utility pole', 'traffic light', 'traffic sign (back)',
                  'traffic sign', 'trash can', 'bicycle', 'boat', 'bus', 'car', 'caravan', 'motorcycle',
                  'on rails', 'other vehicle', 'trailer', 'truck', 'wheeled slow', 'car mount', 'ego vehicle'],
    "camvid": ['bicyclist', 'building', 'car', 'fence', 'person', 'pole', 'road', 'sidewalk', 'sky', 'traffic sign',
               'vegetation'],
    "camvid_cib": ['rider', 'building', 'car', 'fence', 'person', 'pole', 'road', 'sidewalk', 'sky', 'traffic sign',
                   'vegetation'],
    "camvid_v2": ['bicyclist', 'building', 'car', 'fence', 'lane marking - general', 'person', 'pole', 'road',
                  'sidewalk', 'sky', 'traffic sign', 'vegetation'],

    # Obtain by running `get_dataset_category_union`
    "city_idd_bdd_mapillary": ['autorickshaw', 'banner', 'barrier', 'bench', 'bicycle', 'bicyclist', 'bike lane',
                               'bike rack', 'billboard', 'bird', 'boat', 'bridge', 'building', 'bus', 'car',
                               'car mount', 'caravan', 'catch basin', 'cctv camera', 'crosswalk - plain', 'curb',
                               'curb cut', 'ego vehicle', 'fence', 'fire hydrant', 'ground animal', 'guard rail',
                               'junction box', 'lane marking - crosswalk', 'lane marking - general', 'mailbox',
                               'manhole', 'motorcycle', 'motorcyclist', 'mountain', 'obs-str-bar-fallback', 'on rails',
                               'other rider', 'other vehicle', 'parking', 'pedestrian area', 'person', 'phone booth',
                               'pole', 'pothole', 'rail track', 'rider', 'road', 'sand', 'service lane', 'sidewalk',
                               'sky', 'snow', 'street light', 'terrain', 'traffic light', 'traffic sign',
                               'traffic sign (back)', 'traffic sign frame', 'trailer', 'train', 'trash can', 'truck',
                               'tunnel', 'utility pole', 'vegetation', 'vehicle fallback', 'wall', 'water',
                               'wheeled slow'],
    "city_idd_mapillary": ['autorickshaw', 'banner', 'barrier', 'bench', 'bicycle', 'bicyclist', 'bike lane',
                           'bike rack',
                           'billboard', 'bird', 'boat', 'bridge', 'building', 'bus', 'car', 'car mount', 'caravan',
                           'catch basin', 'cctv camera', 'crosswalk - plain', 'curb', 'curb cut', 'ego vehicle',
                           'fence',
                           'fire hydrant', 'ground animal', 'guard rail', 'junction box', 'lane marking - crosswalk',
                           'lane marking - general', 'mailbox', 'manhole', 'motorcycle', 'motorcyclist', 'mountain',
                           'obs-str-bar-fallback', 'on rails', 'other rider', 'other vehicle', 'parking',
                           'pedestrian area',
                           'person', 'phone booth', 'pole', 'pothole', 'rail track', 'rider', 'road', 'sand',
                           'service lane',
                           'sidewalk', 'sky', 'snow', 'street light', 'terrain', 'traffic light', 'traffic sign',
                           'traffic sign (back)', 'traffic sign frame', 'trailer', 'train', 'trash can', 'truck',
                           'tunnel',
                           'utility pole', 'vegetation', 'vehicle fallback', 'wall', 'water', 'wheeled slow'],
    "city_idd_bdd": ['autorickshaw', 'bicycle', 'billboard', 'bridge', 'building', 'bus', 'car', 'curb', 'fence',
                     'guard rail', 'motorcycle', 'obs-str-bar-fallback', 'parking', 'person', 'pole', 'rail track',
                     'rider', 'road', 'sidewalk', 'sky', 'terrain', 'traffic light', 'traffic sign', 'train', 'truck',
                     'vegetation', 'vehicle fallback', 'wall'],
    "idd_bdd_mapillary": ['autorickshaw', 'banner', 'barrier', 'bench', 'bicycle', 'bicyclist', 'bike lane',
                          'bike rack', 'billboard', 'bird', 'boat', 'bridge', 'building', 'bus', 'car', 'car mount',
                          'caravan', 'catch basin', 'cctv camera', 'crosswalk - plain', 'curb', 'curb cut',
                          'ego vehicle', 'fence', 'fire hydrant', 'ground animal', 'guard rail', 'junction box',
                          'lane marking - crosswalk', 'lane marking - general', 'mailbox', 'manhole', 'motorcycle',
                          'motorcyclist', 'mountain', 'obs-str-bar-fallback', 'on rails', 'other rider',
                          'other vehicle', 'parking', 'pedestrian area', 'person', 'phone booth', 'pole', 'pothole',
                          'rail track', 'rider', 'road', 'sand', 'service lane', 'sidewalk', 'sky', 'snow',
                          'street light', 'terrain', 'traffic light', 'traffic sign', 'traffic sign (back)',
                          'traffic sign frame', 'trailer', 'train', 'trash can', 'truck', 'tunnel', 'utility pole',
                          'vegetation', 'vehicle fallback', 'wall', 'water', 'wheeled slow'],
    "city_bdd_mapillary": ['banner', 'barrier', 'bench', 'bicycle', 'bicyclist', 'bike lane', 'bike rack', 'billboard',
                           'bird', 'boat', 'bridge', 'building', 'bus', 'car', 'car mount', 'caravan', 'catch basin',
                           'cctv camera', 'crosswalk - plain', 'curb', 'curb cut', 'ego vehicle', 'fence',
                           'fire hydrant', 'ground animal', 'guard rail', 'junction box', 'lane marking - crosswalk',
                           'lane marking - general', 'mailbox', 'manhole', 'motorcycle', 'motorcyclist', 'mountain',
                           'on rails', 'other rider', 'other vehicle', 'parking', 'pedestrian area', 'person',
                           'phone booth', 'pole', 'pothole', 'rail track', 'rider', 'road', 'sand', 'service lane',
                           'sidewalk', 'sky', 'snow', 'street light', 'terrain', 'traffic light', 'traffic sign',
                           'traffic sign (back)', 'traffic sign frame', 'trailer', 'train', 'trash can', 'truck',
                           'tunnel', 'utility pole', 'vegetation', 'wall', 'water', 'wheeled slow']
}

hierarchies = {
    "city_idd_bdd_mapillary": {'all barrier': ['curb', 'fence', 'wall', 'barrier', 'guard rail'],
                               'animal': ['bird', 'ground animal', 'person', 'motorcyclist', 'bicyclist', 'other rider',
                                          'rider'],
                               'flat': ['road', 'sidewalk', 'curb cut', 'crosswalk - plain', 'parking', 'bike lane',
                                        'service lane', 'rail track', 'pedestrian area', 'lane marking - general',
                                        'lane marking - crosswalk'],
                               'nature': ['sky', 'vegetation', 'terrain', 'mountain', 'snow', 'water', 'sand'],
                               'object': ['street light', 'billboard', 'traffic light', 'manhole', 'banner',
                                          'trash can', 'catch basin', 'junction box', 'cctv camera', 'fire hydrant',
                                          'bench', 'bike rack', 'mailbox', 'pothole', 'phone booth',
                                          'obs-str-bar-fallback'],
                               'structure': ['building', 'bridge', 'tunnel'],
                               'support': ['pole', 'utility pole', 'traffic sign frame'],
                               'traffic sign all': ['traffic sign', 'traffic sign (back)'],
                               'vehicle': ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'other vehicle',
                                           'wheeled slow', 'boat', 'on rails', 'trailer', 'caravan', 'autorickshaw',
                                           'train', 'vehicle fallback'],
                               'void': ['ego vehicle', 'car mount']}
}


def get_hierarchy(dataset_name):
    return hierarchies[dataset_name]


def get_label_2_train(dataset_name):
    return label2train[dataset_name]


def get_dataset_categories(dataset_name):
    return dataset_categories[dataset_name]


def get_dataset_category_union(dataset_names):
    all_dataset_category_lists = [get_dataset_categories(name) for name in dataset_names]
    category_union = sorted(list(set().union(*all_dataset_category_lists)))
    return category_union


def get_combined_ind_mapping(dataset_names: list):
    ind_mappings = {}
    for d_name in dataset_names:
        categories = get_dataset_categories(d_name)
        original_mappings = get_label_2_train(d_name)
        sorted_categories = sorted(categories)
        new_mapping = []
        for m in original_mappings:
            cls_idx = m[1]
            if cls_idx != -1 and cls_idx != 255:
                category_name = categories[cls_idx]
                new_idx = sorted_categories.index(category_name)
                new_m = [m[0], new_idx]
                new_mapping.append(new_m)
            else:
                new_mapping.append(m)
        ind_mappings[d_name] = new_mapping
    return ind_mappings


def get_mixable_categories(dataset_key):
    hierarchy = get_hierarchy(dataset_key)
    if dataset_key == "city_idd_bdd_mapillary":
        mixable_classes = []
        mixable_classes.extend(hierarchy['animal'])
        mixable_classes.extend(hierarchy['object'])
        mixable_classes.extend(hierarchy['vehicle'])
        mixable_classes.extend(hierarchy['traffic sign all'])
        mixable_classes.extend(hierarchy['support'])
    else:
        raise NotImplementedError("No implementation for {}".format(dataset_key))
    return mixable_classes


def get_combined_dataset_label2train(dataset_names: list):
    """
    Returns new mappings from original mask pixel values to class labels when all categories from
    each dataset_configs in `dataset_names` is unionized.
    """
    category_union = get_dataset_category_union(dataset_names)
    new_label2train = {}
    for d_name in dataset_names:
        category_names = get_dataset_categories(d_name)
        label2train = get_label_2_train(d_name)
        label2train = np.array(label2train)
        modified_label2train = np.copy(label2train)
        for i, cat_name in enumerate(category_names):
            cat_idx_in_union = category_union.index(cat_name)
            modified_label2train[label2train[:, 1] == i] = cat_idx_in_union
            modified_label2train[:, 0] = label2train[:, 0]
        new_label2train[d_name] = modified_label2train.tolist()
    return new_label2train


def label_mapping(input_mask, mapping):
    output = np.copy(input_mask)
    for idx in range(len(mapping)):
        output[input_mask == mapping[idx][0]] = mapping[idx][1]
    return np.array(output, dtype=np.int32)
