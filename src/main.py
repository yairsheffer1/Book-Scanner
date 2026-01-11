import cornerCrop
import stableFrames
import arrange
import matplotlib
import pageSplit
import pageGrouping
import cv2
import zucker_dewarp
import fix_folder


num_of_pages = 9
path = r"C:\Users\user\Desktop\project\9.M4V"

def run_full_processing(path, num_of_pages):
    matplotlib.use('TkAgg')

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("step A")
    stableFrames.video_to_images(path, num_of_pages)


    print("step B")
    pageSplit.folder_split(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Motion_detection_S1")
    pageSplit.delete_outliers_pages(
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Page_split_S2",
        side='left', tolerance=0.10)
    pageSplit.delete_outliers_pages(
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Page_split_S2",
        side='right', tolerance=0.10)


    print("step C")
    cornerCrop.analyze_all_crop(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Page_split_S2")


    print("step D")
    (filtered_L, filtered_R, matching_pairs, dist_matrix_L, dist_matrix_R, image_paths_L,
     image_paths_R) = pageGrouping.run_full_processing(
    r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_R_S3",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_L_S3",
        num_of_pages,
        total_frames
    )


    print("step D - Plotting Clustering Visualization")
    # page_compare.plot_distance_matrix_and_dendrogram(dist_matrix_L, image_paths_L, title="Left Pages Clustering")
    # page_compare.plot_distance_matrix_and_dendrogram(dist_matrix_R, image_paths_R, title="Right Pages Clustering")

    #
    print("step E")
    arrange.create_multiple_pdfs_with_filters(r"C:\Users\user\PycharmProjects\pythonProject10\results\merged_book",
        filtered_L,
        filtered_R,
        matching_pairs)
    arrange.apply_masks_to_folder(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_L_S4",
                             r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\L")
    arrange.apply_masks_to_folder(
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_R_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\R")

    print("step F")
    zucker_dewarp.run1(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\L",
                       r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_L_S4")
    zucker_dewarp.run1(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\R",
                       r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_R_S4")


    zucker_dewarp.run0(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf\L",
                       r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_L_S4")
    zucker_dewarp.run0(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf\R",
                       r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_R_S4")
    #
    fix_folder.sync_folders_by_number(
        source_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf\L",
        target_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_L_S4"
    )
    fix_folder.sync_folders_by_number(
        source_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf\R",
        target_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_R_S4"
    )
    fix_folder.sync_folders_by_number(
        source_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\L",
        target_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_L_S4"
    )
    fix_folder.sync_folders_by_number(
        source_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\R",
        target_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_R_S4"
    )
    arrange.create_interleaved_pdf_by_number(
    right_folder=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_R_S4",
    left_folder=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_L_S4",
    output_pdf_path=r"C:\Users\user\PycharmProjects\pythonProject10\results\Scanned_Book_V1.pdf"
    )
    arrange.create_interleaved_pdf_by_number(
        right_folder=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_R_S4",
        left_folder=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_L_S4",
        output_pdf_path=r"C:\Users\user\PycharmProjects\pythonProject10\results\Scanned_Book_V2.pdf"
    )
    print("END!!!")

def erase():
    import os
    folders = [
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_L_S3",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_R_S3",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_L_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_R_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Motion_detection_S1",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Page_split_S2",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_L_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals_Dewarped_R_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_L_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\finals1_Dewarped_R_S4",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\L",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf1\R",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf\L",
        r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\filesOfPdf\R"
    ]

    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")



run_full_processing(path, num_of_pages)


