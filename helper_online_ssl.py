import cv2 as cv
import os
import numpy as np
from scipy.spatial import distance

face_haar_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_haar_cascade = cv.CascadeClassifier("data/haarcascade_eye.xml")


def online_face_recognition(profile_names,
                            IncrementalKCenters,
                            n_pictures=15,
                            video_filename=None):
    """
    Run online face recognition.

    Parameters
    ----------
    profile_names : list
        List of user names used in create_user_profile()
    IncrementalKCenters : class
        Class implementing Incremental k-centers
    n_pictures : int
        Number of (labeled) pictures to use for each user_name
    video_filename : str
        .mp4 video file. If None, read from camera
    """
    images = []
    labels = []
    label_names = []
    for i, name in enumerate(profile_names):
        p = load_profile(name)
        p = p[0:n_pictures, ]
        images += [p]
        labels += [np.ones(p.shape[0]) * (i + 1)]
        label_names += [name]
    faces = np.vstack(images)
    labels = np.hstack(labels).astype(np.int64)
    #  Generate model
    model = IncrementalKCenters(faces, labels, label_names)
    # Start camera
    if video_filename is None:
        cam = cv.VideoCapture(0)
    else:
        cam = cv.VideoCapture(video_filename)
    while True:
        ret_val, img = cam.read()
        working_image, grey_image = preprocess_camera_image(img)
        box = face_haar_cascade.detectMultiScale(working_image)
        for b0 in box:
            x, y = b0[0], b0[1]
            x_range, y_range = b0[2], b0[3]
            # look for eye classifier
            local_image = img[y:(y + y_range), x:(x + x_range)]
            eye_box = eye_haar_cascade.detectMultiScale(local_image)
            if len(eye_box) == 0:
                cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]),
                             (0, 0, 255), 2)
                continue
            # select face
            local_image = grey_image[y:(y + y_range), x:(x + x_range)]
            x_t = preprocess_face(local_image)

            """
            Centroids are updated here
            """
            model.online_ssl_update_centroids(x_t)
            p1, p2 = tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4])

            """
            Hard HFS solution is computed here
            """
            label_scores = model.online_ssl_compute_solution()
            scores = [ll[1] for ll in label_scores]
            labels = [ll[0] for ll in label_scores]
            sorted_label_indices = np.argsort(scores)

            """
            Show results
            """
            for ii, ll_idx in enumerate(sorted_label_indices):
                label = labels[ll_idx]
                score = scores[ll_idx]
                if label not in label_names:
                    color = (100, 100, 100)
                else:
                    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][ll_idx % 3]
                txt = label + "  " + ('%.4f' % score)
                cv.putText(img, txt, (p1[0], p1[1] - 5 - 10 * ii), cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                0.5 + 0.5 * (ii == len(scores) - 1), color)

            cv.rectangle(img, p1, p2, color, 2)
        cv.putText(img, "Face recognition: [s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]:
            break
        if key == ord('s'):
            # Save face
            print('saved')
            # cv.imwrite("frame.png", img)
            if not os.path.exists('results'):
                os.makedirs('results')
            image_name = os.path.join('results', 'frame.png')
            cv.imwrite(image_name, img)
            print("Image saved at", image_name)

            ## cv.waitKey(1)
    cv.destroyAllWindows()


def increasing_online_face_recognition(profile_names,
                            IncrementalKCenters,
                            n_pictures=15,
                            max_centroids=50,
                            acc_threshold=0.5,
                            unknown_threshold=0.7,
                            video_filename=None):
    """
    Run online face recognition. If a new face is detected, it is added to the model.

    Parameters
    ----------
    profile_names : list
        List of user names used in create_user_profile()
    IncrementalKCenters : class
        Class implementing Incremental k-centers
    max_centroids : int
        Maximum number of centroids
    acc_threshold : float
        Threshold for accepting a new face as a potential new user
    n_pictures : int
        Number of (labeled) pictures to use for each user_name
    unknown_threshold : float
        Threshold for unknown faces
    video_filename : str
        .mp4 video file. If None, read from camera
    """
    images = []
    labels = []
    label_names = []
    for i, name in enumerate(profile_names):
        p = load_profile(name)
        p = p[0:n_pictures, ]
        images += [p]
        labels += [np.ones(p.shape[0]) * (i + 1)]
        label_names += [name]
    faces = np.vstack(images)
    labels = np.hstack(labels).astype(np.int64)
    #  Generate model
    model = IncrementalKCenters(faces, labels, label_names, max_num_centroids=max_centroids)
    # Start camera
    if video_filename is None:
        cam = cv.VideoCapture(0)
    else:
        cam = cv.VideoCapture(video_filename)
    # Unknown face :
    unknown_face = {}
    unknown_face['data'] = []
    unknown_face['indice'] = []
    while True:
        ret_val, img = cam.read()
        working_image, grey_image = preprocess_camera_image(img)
        box = face_haar_cascade.detectMultiScale(working_image)
        for b0 in box:
            x, y = b0[0], b0[1]
            x_range, y_range = b0[2], b0[3]
            # look for eye classifier
            local_image = img[y:(y + y_range), x:(x + x_range)]
            eye_box = eye_haar_cascade.detectMultiScale(local_image)
            if len(eye_box) == 0:
                cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]),
                             (0, 0, 255), 2)
                continue
            # select face
            local_image = grey_image[y:(y + y_range), x:(x + x_range)]
            x_t = preprocess_face(local_image)

            """
            Centroids are updated here
            """
            model.online_ssl_update_centroids(x_t)
            p1, p2 = tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4])

            """
            Hard HFS solution is computed here
            """
            label_scores = model.online_ssl_compute_solution(unknown_threshold=unknown_threshold)
            scores = [ll[1] for ll in label_scores]
            labels = [ll[0] for ll in label_scores]
            sorted_label_indices = np.argsort(scores)

            """
            Handling unknown faces
            """
            if len(scores) == 1:
                unknown_face['data'].append(x_t)
                unknown_face['indice'].append(len(model.centroids)-1)
            
            if len(unknown_face['data']) > n_pictures:
                # print(len(unknown_face['data']), "unknown faces detected")
                unlabeled_centroid_idx = np.where(model.Y == 0)[0]
                if len(unlabeled_centroid_idx) == 0:
                    print("No unlabeled centroid found - increase max_centroids to get more mathemathicians recognized")
                else :
                    distances = distance.cdist(np.array(unknown_face['data']), model.centroids[unlabeled_centroid_idx], 'euclidean')
                    valid = np.array([])
                    UU = 0
                    new_valid = np.array([])
                    len_valid = 0
                    for uu in range(len(unlabeled_centroid_idx)):
                        distances[:, uu] = (distances[:, uu] - np.min(distances[:, uu])) / (np.percentile(distances[:, uu], 95) - np.min(distances[:, uu]))
                        new_valid = np.where(distances[:, uu] < acc_threshold)[0]
                        if len(new_valid) > len_valid:
                            valid = new_valid
                            UU = uu
                            len_valid = len(new_valid)
                    if len(valid) >= n_pictures:
                        new_class = np.max(model.Y) + 1
                        new_label = 'mathematician n°' + str(new_class - len(profile_names))
                        model.Y[np.array(unknown_face['indice'])[valid]] = new_class
                        model.Y[unlabeled_centroid_idx[UU]] = new_class
                        unlabeled_centroid_distances = distance.cdist(model.centroids[unlabeled_centroid_idx], model.centroids[unlabeled_centroid_idx], 'euclidean')[:, UU]
                        unlabeled_centroid_distances = (unlabeled_centroid_distances - np.min(unlabeled_centroid_distances)) / (np.percentile(unlabeled_centroid_distances, 98) - np.min(unlabeled_centroid_distances))
                        labelable_centroid = np.where(unlabeled_centroid_distances < acc_threshold)[0]
                        if len(labelable_centroid) > 0:
                            model.Y[unlabeled_centroid_idx[labelable_centroid]] = new_class
                        model.label_names += [new_label]
                        print("New mathematician recognized : " + new_label)
                        # reset all the data similar to the new mathemathician
                        face_similarity_distance = distance.cdist(np.array(unknown_face['data']), np.array(unknown_face['data'])[valid], 'euclidean')
                        face_similarity_distance = (face_similarity_distance - np.min(face_similarity_distance)) / (np.max(face_similarity_distance) - np.min(face_similarity_distance))
                        # min distance along the second axis
                        face_similarity_distance = np.min(face_similarity_distance, axis=1)
                        valid = np.where(face_similarity_distance < 0.6)[0]
                        unknown_face['data'] = [x for i, x in enumerate(unknown_face['data']) if i not in valid]
                        unknown_face['indice'] = [x for i, x in enumerate(unknown_face['indice']) if i not in valid]

            """
            Show results
            """
            for ii, ll_idx in enumerate(sorted_label_indices):
                label = labels[ll_idx]
                score = scores[ll_idx]
                if label not in label_names:
                    color = (100, 100, 100)
                else:
                    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][ll_idx % 3]
                txt = label + "  " + ('%.4f' % score)
                cv.putText(img, txt, (p1[0], p1[1] - 5 - 10 * ii), cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                0.5 + 0.5 * (ii == len(scores) - 1), color)

            cv.rectangle(img, p1, p2, color, 2)
        cv.putText(img, "Face recognition: [s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]:
            break
        if key == ord('s'):
            # Save face
            print('saved')
            # cv.imwrite("frame.png", img)
            if not os.path.exists('results'):
                os.makedirs('results')
            image_name = os.path.join('results', 'frame.png')
            cv.imwrite(image_name, img)
            print("Image saved at", image_name)

            ## cv.waitKey(1)
    cv.destroyAllWindows()



def create_user_profile(user_name,
                        faces_path="data/",
                        video_filename=None):
    """
    Uses the camera to collect data.
    
    Parameters
    ----------
    user_name : str
        Name that identifies the person/face
    faces_path : str
        Where to store the images
    video_filename : str
        .mp4 video file. If None, read from camera
    """
    # Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    image_count = 0
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
        print("New profile created at path", profile_path)
    else:
        image_count = len(os.listdir(profile_path))
        print("Profile found with", image_count, "images.")
    # Launch video capture
    if video_filename is None:
        cam = cv.VideoCapture(0)
    else:
        cam = cv.VideoCapture(video_filename)
        print("Reading from video file", video_filename)
    while True:
        ret_val, img = cam.read()
        working_image, grey_image = preprocess_camera_image(img)
        box = face_haar_cascade.detectMultiScale(working_image)
        if len(box) > 0:
            box_surface = box[:, 2] * box[:, 3]
            index = box_surface.argmax()
            b0 = box[index]
            cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]), (0, 255, 0),
                         2)
        cv.putText(img, f"Create profile ({user_name}): [s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]: break  # esc or e to quit
        if key == ord('s'):
            ## Save face
            if len(box) > 0:
                x, y = b0[0], b0[1]
                x_range, y_range = b0[2], b0[3]
                image_count = image_count + 1
                image_name = os.path.join(profile_path, "img_" + str(image_count) + ".bmp")
                img_to_save = img[y:(y + y_range), x:(x + x_range)]
                cv.imwrite(image_name, img_to_save)
                print("Image", image_count, "saved at", image_name)
    cv.destroyAllWindows()
    return


def load_profile(user_name, faces_path="data/"):
    """
    Loads the data associated to user_name.

    Returns an array of shape (number_of_images, n_pixels)
    """
    assert ("faces" in os.listdir(faces_path)), "Error : 'faces' folder not found"
    ## Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    if not os.path.exists(profile_path):
        raise Exception("Profile not found")
    image_count = len(os.listdir(profile_path))
    print("Profile found with", image_count, "images.")
    images = [os.path.join(profile_path, x) for x in os.listdir(profile_path)]
    rep = np.zeros((len(images), 96 * 96))
    for i, im_path in enumerate(images):
        im = cv.imread(im_path, 0)
        cv.waitKey(1)
        rep[i, :] = preprocess_face(im)
    return rep


def preprocess_camera_image(img):
    """
    Preprocessing for face detection
    """
    grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    working_image = cv.bilateralFilter(grey_image, 9, 75, 75)
    working_image = cv.equalizeHist(working_image)
    working_image = cv.GaussianBlur(working_image, (5, 5), 0)
    return working_image, grey_image


def preprocess_face(grey_face):
    """
    Transforms a n x n image into a feature vector
    :param grey_face: ( n x n ) image in grayscale
    :return gray_face_vector:  ( 1 x EXTR_FRAME_SIZE^2) row vector with the preprocessed face
    """
    # Face preprocessing
    EXTR_FRAME_SIZE = 96
    """
     Apply preprocessing to balance the image (color/lightning), such    
      as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and 
      equalization (cv.equalizeHist).                                     
    """
    grey_face = cv.bilateralFilter(grey_face, 9, 75, 75)
    grey_face = cv.equalizeHist(grey_face)
    grey_face = cv.GaussianBlur(grey_face, (5, 5), 0)

    # resize the face
    grey_face = cv.resize(grey_face, (EXTR_FRAME_SIZE, EXTR_FRAME_SIZE))
    grey_face = grey_face.reshape(EXTR_FRAME_SIZE * EXTR_FRAME_SIZE).astype(np.float64)
    grey_face -= grey_face.mean()
    grey_face /= grey_face.max()

    return grey_face
