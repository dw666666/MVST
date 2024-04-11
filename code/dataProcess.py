import scanpy as sc
import pandas as pd
import os
from skimage import io, img_as_float32, morphology, exposure
import skimage
from skimage.feature import greycomatrix, greycoprops
from itertools import product
from sklearn.preprocessing import minmax_scale
from skimage.color import separate_stains, hdx_from_rgb
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing


def extract_img_features(
        input_path,
        input_type,
        output_path,
        img=None,
        img_meta=None,
        feature_mask_shape="spot",
):

    intensity_fn = os.path.join(
        os.path.abspath(output_path),
        "{}_level_texture_features.csv".format(feature_mask_shape)
    )
    texture_fn = os.path.join(
        os.path.abspath(output_path),
        "{}_level_intensity_features.csv".format(feature_mask_shape)
    )
    if (os.path.exists(intensity_fn)) == (os.path.exists(texture_fn)) == True:
        print('Features are already extracted.')
        return
    if img_meta is None:
        img_meta = pd.read_csv(
            os.path.join(input_path, "Spot_metadata.csv"), index_col=0)
    if img is None:
        img_tif = [x for x in os.listdir(input_path) if "tif" in x][0]
        img_tif = os.path.join(input_path, img_tif)
        if input_type == "if":
            img = io.imread(img_tif)
            img = img_as_float32(img)
            img = (255 * img).astype("uint8")
        else:
            img = io.imread(img_tif)

            # normalize image with color deconv
            print('Normalizing image...')
            img = separate_stains(img, hdx_from_rgb)
            print("1")
            img = minmax_scale(img.reshape(-1, 3)).reshape(img.shape)
            print("2")
            img = np.clip(img, 0, 1)
            print("3")
            img = img.astype(np.float32)
            img = exposure.equalize_adapthist(img, clip_limit=0.01)
            img = (255 * img).astype("uint8")

    if feature_mask_shape == "block":
        tmp = img_meta.sort_values(["Row", "Col"])
        block_y = int(np.median(tmp.Y.values[2:-1] - tmp.Y.values[1:-2]) // 2)
        tmp = img_meta.sort_values(["Col", "Row"])
        block_x = int(np.median(tmp.X.values[2:-1] - tmp.X.values[1:-2]) // 2)
        block_r = min(block_x, block_y)
        block_x = block_y = block_r
    print("Prossessing {}".format(input_path))
    feature_set = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "ASM",
        "energy",
        "correlation",
    ]
    text_features = []
    intensity_features = []
    for i in range(img_meta.shape[0]):
        if (i + 1) % 100 == 0:
            print("Processing {} spot out of {} spots".format(i + 1, img_meta.shape[0]))
        row = img_meta.iloc[i]
        x, y, r = row[["X", "Y", "Spot_radius"]].astype(int)
        if feature_mask_shape == "spot":
            spot_img = img[x - r: x + r + 1, y - r: y + r + 1]
            spot_mask = morphology.disk(r)

            spot_img = np.einsum("ij,ijk->ijk", spot_mask, spot_img)
        else:
            spot_img = img[x - block_x: x + block_x + 1, y - block_y: y + block_y + 1]
            spot_mask = np.ones_like(spot_img[:, :, 0], dtype="bool")

        # extract texture features
        ith_texture_f = []
        for c in range(img.shape[2]):
            glcm = skimage.feature.greycomatrix(
                spot_img[:, :, c],
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=False,
            )

            glcm = glcm[1:, 1:]
            glcm = glcm / np.sum(glcm, axis=(0, 1))
            for feature_name in feature_set:
                ith_texture_f += skimage.feature.graycoprops(glcm, feature_name)[0].tolist()
    text_features.append(ith_texture_f)

    # extract intensity features
    int_low = 0.2
    int_high = 0.8
    int_step = 0.1
    q_bins = np.arange(int_low, int_high, int_step)
    ith_int_f = []
    for c in range(img.shape[2]):
        for t in q_bins:
            ith_int_f.append(np.quantile(spot_img[:, :, c][spot_mask == True], t))
    intensity_features.append(ith_int_f)
    channels = ["f" + str(i) for i in range(img.shape[2])]
    col_names = product(channels, feature_set, ["A1", "A2", "A3", "A4"])
    col_names = ["_".join(x) for x in col_names]
    text_features = pd.DataFrame(text_features, index=img_meta.index, columns=col_names)
    # construct intensity feature table
    intensity_features = pd.DataFrame(
        intensity_features,
        index=img_meta.index,
        columns=[
            "_".join(x) for x in product(channels, ["{:.1f}".format(x) for x in q_bins])
        ],
    )
    text_features.to_csv(intensity_fn)
    intensity_features.to_csv(texture_fn)
    return text_features, intensity_features


def get_type(data_name, cell_types, generated_data_fold):
    types_dic = []
    types_idx = []
    for t in cell_types:
        if not t in types_dic:
            types_dic.append(t)
        id = types_dic.index(t)
        types_idx.append(id)

    n_types = max(types_idx) + 1
    if data_name == 'V1_Breast_Cancer_Block_A_Section_1':
        types_dic_sorted = ['Healthy_1', 'Healthy_2', 'Tumor_edge_1', 'Tumor_edge_2', 'Tumor_edge_3', 'Tumor_edge_4',
                            'Tumor_edge_5', 'Tumor_edge_6',
                            'DCIS/LCIS_1', 'DCIS/LCIS_2', 'DCIS/LCIS_3', 'DCIS/LCIS_4', 'DCIS/LCIS_5', 'IDC_1', 'IDC_2',
                            'IDC_3', 'IDC_4', 'IDC_5', 'IDC_6', 'IDC_7']
        relabel_map = {}
        cell_types_relabel = []
        for i in range(n_types):
            relabel_map[i] = types_dic_sorted.index(types_dic[i])
        for old_index in types_idx:
            cell_types_relabel.append(relabel_map[old_index])

        np.save(generated_data_fold + 'cell_types.npy', np.array(cell_types_relabel))
        np.savetxt(generated_data_fold + 'types_dic.txt', np.array(types_dic_sorted), fmt='%s', delimiter='\t')
    elif data_name == 'DLPFC151509':
        types_dic_sorted = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6',
                            'WM']
        relabel_map = {}
        cell_types_relabel = []
        for i in range(n_types):
            relabel_map[i] = types_dic_sorted.index(types_dic[i])
        for old_index in types_idx:
            cell_types_relabel.append(relabel_map[old_index])

        np.save(generated_data_fold + 'cell_types.npy', np.array(cell_types_relabel))
        np.savetxt(generated_data_fold + 'types_dic.txt', np.array(types_dic_sorted), fmt='%s', delimiter='\t')
    elif data_name == 'Mouse_Brain_Anterior':
        types_dic_sorted = ['AcbC', 'AcbSh', 'AOB::Gl', 'AOB::Gr', 'AOB::Ml', 'AOE', 'AON::L1_1', 'AON::L1_2',
                            'AON::L2', 'CC', 'Cl', 'CPu',
                            'En', 'Fim', 'FRP::L1', 'FRP::L2/3', 'Ft', 'HY::LPO', 'Io', 'LV', 'MO::L1', 'MO::L2/3',
                            'MO::L5', 'MO::L6', 'MOB::Gl_1', 'MOB::Gl_2', 'MOB::Gr',
                            'MOB::lpl', 'MOB::MI', 'MOB::Opl', 'Not_annotated', 'Or', 'ORB::L1', 'ORB::L2/3', 'ORB::L5',
                            'ORB::L6', 'OT::Ml', 'OT::Pl', 'OT::PoL',
                            'Pal::GPi', 'Pal::MA', 'Pal::NDB', 'Pal::Sl', 'PIR', 'Py', 'SLu', 'SS::L1', 'SS::L2/3',
                            'SS::L5', 'SS::L6', 'St', 'TH::RT']
        relabel_map = {}
        cell_types_relabel = []
        for i in range(n_types):
            relabel_map[i] = types_dic_sorted.index(types_dic[i])
        for old_index in types_idx:
            cell_types_relabel.append(relabel_map[old_index])

        np.save(generated_data_fold + 'cell_types.npy', np.array(cell_types_relabel))
        np.savetxt(generated_data_fold + 'types_dic.txt', np.array(types_dic_sorted), fmt='%s', delimiter='\t')
    else:
        np.save(generated_data_fold + 'cell_types.npy', np.array(cell_types))
        np.savetxt(generated_data_fold + 'types_dic.txt', np.array(types_dic), fmt='%s', delimiter='\t')


def dataProcess(args):
    # Set file location information

    data_fold = args.data_path + args.data_name + '/'
    preprocessed_data_fold = args.preprocessed_data_path + args.data_name + '/'
    data_spatial = data_fold + 'spatial/'
    if not os.path.exists(args.preprocessed_data_path):
        os.makedirs(args.preprocessed_data_path)
    if not os.path.exists(preprocessed_data_fold):
        os.makedirs(preprocessed_data_fold)
    if not os.path.exists(preprocessed_data_fold + 'input/'):
        os.makedirs(preprocessed_data_fold + 'input/')

    # if args.Data_name == "DLPFC":
    #     positions = pd.read_csv(data_fold + 'spatial/tissue_positions_list.txt', delimiter=',',
    #                             header=None)
    #     positions.to_csv(data_spatial + 'tissue_positions_list.csv', index=False, header=None)

    # read data
    if args.Data_name == "DLPFC":
        adata = sc.read_visium(path=data_fold, count_file=args.data_name + '_filtered_feature_bc_matrix.h5')
        metadata_notNa = pd.read_csv(args.data_path + args.data_name + "/cluster_labels_"+args.DLPFC+".csv")
        label_notNa = pd.Categorical(metadata_notNa['ground_truth']).codes
        adata = adata[label_notNa != -1]
    else:
        adata = sc.read_visium(path=data_fold, count_file=args.data_name + '_filtered_feature_bc_matrix.h5')

    spot_meta_ = adata.obsm['spatial']
    np.save(preprocessed_data_fold + 'input/' + 'coordinates.npy', np.array(spot_meta_))
    spot_meta = pd.DataFrame(spot_meta_)
    spot_meta.columns = ['X', 'Y']
    # ifDLPFC
    if args.Data_name == "DLPFC":
        radius = round(adata.uns['spatial'][args.DLPFC]['scalefactors']['spot_diameter_fullres'] / 2, 4)
    else:
        radius = round(adata.uns['spatial'][args.Data_name]['scalefactors']['spot_diameter_fullres'] / 2, 4)
    spot_meta['Spot_radius'] = radius
    spot_meta.to_csv(data_fold + 'Spot_metadata.csv')

    adata.var_names_make_unique()
    # Quality control filtering
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    sc.pp.normalize_total(adata, target_sum=1, inplace=False)
    sc.pp.log1p(adata)
    if args.highly_variable_or == 1:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=args.highly_variable)
        adata_HVG = adata[:, adata.var.highly_variable]
        adata_HVG_X = adata_HVG.X
    else:
        adata_HVG = adata.X.copy()
        adata_HVG_X = adata_HVG.toarray()

    # Data scaling
    adata_HVG_X = sc.pp.scale(adata_HVG_X)

    # Dimensionality reduction and preservation of cellular genetic features
    pca = PCA(n_components=args.Dim_PCA)
    pca_data = pca.fit_transform(adata_HVG_X)
    inputData = preprocessed_data_fold + 'input/'
    if not os.path.exists(inputData):
        os.makedirs(inputData)
    np.save(inputData + "features_" + str(args.highly_variable) + "_PCA.npy", pca_data)

    # Organizational image processing
    image_fold = args.preprocessed_data_path + args.data_name + "/组织图像处理/"
    spot_level_intensity = image_fold + "spot_level_intensity_features.csv"
    if not os.path.exists(image_fold):
        os.makedirs(image_fold)

    if not os.path.exists(spot_level_intensity):
        _ = extract_img_features(args.data_path + args.data_name + "/", "he", image_fold)

    # Coordinate network, transcriptome similarity network, and tissue image similarity network
    nbrs_xy = NearestNeighbors(n_neighbors=args.k + 1, algorithm='auto').fit(adata.obsm['spatial'])
    distances_xy, indices_xy = nbrs_xy.kneighbors(adata.obsm['spatial'])
    A1_dict = {}
    for i in range(len(indices_xy)):
        A1_dict[i] = indices_xy[i][1:args.k + 1].tolist()
    f = open(inputData + 'A1_' + str(args.k) + '_dict.txt', 'w')
    f.write(str(A1_dict))
    f.close()

    pca_ = PCA(n_components=args.Dim_PCA_net)
    pcaData = pca_.fit_transform(adata_HVG_X)
    nbrs = NearestNeighbors(n_neighbors=args.k + 1, algorithm='ball_tree').fit(pcaData)
    distances, indices = nbrs.kneighbors(pcaData)
    A2_dict = {}
    for i in range(len(indices)):
        A2_dict[i] = indices[i][1:args.k + 1].tolist()
    f = open(inputData + 'A2_' + str(args.highly_variable) + '_' + str(args.k) + '_dict.txt', 'w')
    f.write(str(A2_dict))
    f.close()

    pic_feature1 = pd.read_csv(
        args.preprocessed_data_path + args.data_name + "/组织图像处理/spot_level_intensity_features.csv", index_col=0)
    pic_feature2 = pd.read_csv(
        args.preprocessed_data_path + args.data_name + "/组织图像处理/spot_level_texture_features.csv", index_col=0)
    pic_feature = pd.concat([pic_feature1, pic_feature2], axis=1)
    # Robust standardization
    robust = preprocessing.RobustScaler()
    pic_feature_robust = robust.fit_transform(pic_feature)
    pic_feature_robust = pd.DataFrame(pic_feature_robust, index=pic_feature.index, columns=pic_feature.columns)
    pic_feature_robust = np.array(pic_feature_robust)

    # Determine if there is a NAN
    np.argwhere(np.isnan(pic_feature_robust))
    pic_feature_robust[np.isnan(pic_feature_robust)] = 0

    nbrs_pic = NearestNeighbors(n_neighbors=args.k + 1, algorithm='ball_tree').fit(pic_feature_robust)
    distances_pic, indices_pic = nbrs_pic.kneighbors(pic_feature_robust)
    A3_dict = {}
    for i in range(len(indices_pic)):
        A3_dict[i] = indices_pic[i][1:args.k + 1].tolist()
    f = open(inputData + 'A3_' + str(args.k) + '_dict.txt', 'w')
    f.write(str(A3_dict))
    f.close()

    # Read label
    # If there is NA in DLPFC, replace it
    if args.Data_name == 'DLPFC':
        metadata = pd.read_csv(args.data_path + args.data_name + "/cluster_labels_" + args.DLPFC + ".csv")
        label = pd.Categorical(metadata['ground_truth']).codes
        label = label[label != -1]

        metadata.fillna('N____A', inplace=True)
        metadata_type = metadata['ground_truth']
        metadata_type = metadata_type[metadata_type != 'N____A']
        get_type(args.data_name+args.DLPFC, metadata_type,inputData)
        np.save(inputData + "label.npy", label)
    else:
        if args.Data_name == 'V1_Mouse_Brain_Sagittal_Anterior':
            metadata = pd.read_csv(args.data_path + args.data_name + "/metadata.tsv", sep="\t")
            metadata.fillna('N____A', inplace=True)  # Fill empty values in DataFrame with 0
            metadata = metadata.rename(columns={'ground_truth': 'fine_annot_type'})
            metadata_list = metadata.iloc[:, 9].tolist()
        else:
            metadata = pd.read_csv(args.data_path + args.data_name + "/metadata.tsv", sep="\t")
            metadata_list = metadata.iloc[:, 2].tolist()

        metadata_list = list(set(metadata_list))
        print(len(metadata_list))
        size_mapping = {}
        size_mapping_idx = 0
        for i in metadata_list:
            size_mapping[i] = size_mapping_idx
            size_mapping_idx += 1
        get_type(args.Data_name, metadata['fine_annot_type'], inputData)
        metadata['fine_annot_type'] = metadata['fine_annot_type'].map(size_mapping)
        metadata_array = metadata['fine_annot_type'].to_numpy()
        np.save(inputData + "label.npy", metadata_array)

    if args.clusterOr == True:
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors)
        sc.tl.louvain(adata, resolution=args.res)
        y_pred = adata.obs['louvain'].astype(int).to_numpy()
        n = len(np.unique(y_pred))
        n = np.array([n])
        np.savetxt(inputData + 'cluster.txt', n, delimiter=',')
















