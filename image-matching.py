# %%
import cv2
import matplotlib.pyplot as plt


# %%
def resize(img, max_size):
    """Utility function for image resizing"""
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scaling_factor = max_size / float(max(height, width))
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
    return img


# %%
def preprocessing(img):
    img = resize(img, 1600)
    return img


def load_images(name):
    img1 = cv2.imread(f"images/{name}1.JPG", cv2.IMREAD_GRAYSCALE)
    img1 = preprocessing(img1)
    img2 = cv2.imread(f"images/{name}2.JPG", cv2.IMREAD_GRAYSCALE)
    img2 = preprocessing(img2)

    return img1, img2


# %%


def match(img1, img2, f_det_dscrpt, ratio=0.75):
    # Find the keypoints and descriptors with the given feature detector
    kp1, des1 = f_det_dscrpt.detectAndCompute(img1, None)
    kp2, des2 = f_det_dscrpt.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher (Brute-Force Matcher)
    brute_force = cv2.BFMatcher()
    matches = brute_force.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    # Draw matching result
    img_matches = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return img_matches


# %%
def sift_detect_factory():
    sift = cv2.SIFT_create()
    return lambda img, mask=None: sift.detectAndCompute(img, mask)


def orb_detect_factory():
    orb = cv2.ORB_create()
    return lambda img, mask=None: orb.detectAndCompute(img, mask)


def akaze_detect_factory():
    akaze = cv2.AKAZE_create()
    return lambda img, mask=None: akaze.detectAndCompute(img, mask)


def kaze_detect_factory():
    kaze = cv2.KAZE_create()
    return lambda img, mask=None: kaze.detectAndCompute(img, mask)


def save_and_show_matching_results(name):
    img1, img2 = load_images(name)

    img_matches_sift = match(img1, img2, cv2.SIFT_create(), ratio=0.75)
    img_matches_orb = match(img1, img2, cv2.ORB_create(), ratio=0.75)
    img_matches_kaze = match(img1, img2, cv2.KAZE_create(), ratio=0.75)
    img_matches_akaze = match(img1, img2, cv2.AKAZE_create(), ratio=0.75)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(name)
    axs = axs.flatten()

    axs[0].imshow(img_matches_sift)
    axs[0].set_title("SIFT")

    axs[1].imshow(img_matches_orb)
    axs[1].set_title("ORB")

    axs[2].imshow(img_matches_kaze)
    axs[2].set_title("KAZE")

    axs[3].imshow(img_matches_akaze)
    axs[3].set_title("AKAZE")
    fig.savefig(f"results/{name}_matching.png")

    return fig


def main():
    # %%
    fig = save_and_show_matching_results("piano")
    fig.show()

    # %%
    fig = save_and_show_matching_results("car")
    fig.show()

    # %%
    fig = save_and_show_matching_results("korankei")
    fig.show()

    # %%
    fig = save_and_show_matching_results("tut")
    fig.show()


if __name__ == "__main__":
    main()
