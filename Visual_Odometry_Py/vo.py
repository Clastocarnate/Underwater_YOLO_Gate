import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


class VisualOdometry():
    def __init__(self):
        self.K = np.array([[964.91351658, 0, 644.64335809],
                           [0, 965.0282362, 353.26976819],
                           [0, 0, 1]])
        self.P = np.hstack((self.K, np.zeros((3, 1))))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            exit()
        self.prev_frame = None
        self.cur_pose = np.eye(4)  # Initial pose as the identity matrix

        # Initialize plot
        self.x_coords = [0]
        self.z_coords = [0]
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.x_coords, self.z_coords, '-o')
        self.ax.set_xlabel('X (Forward)')
        self.ax.set_ylabel('Z (Sideways)')
        self.ax.set_title('Live Trajectory')

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, prev_frame, curr_frame):
        kp1, des1 = self.orb.detectAndCompute(prev_frame, None)
        kp2, des2 = self.orb.detectAndCompute(curr_frame, None)

        if des1 is None or des2 is None:
            return np.array([]), np.array([])  # Return empty arrays if no descriptors are found

        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:  # Ensure there are two matches to unpack
                m, n = m_n
                if m.distance < 0.8 * n.distance:
                    good.append(m)

        img3 = cv2.drawMatches(curr_frame, kp1, prev_frame, kp2, good, None, flags=2)
        cv2.imshow("Matches", img3)
        cv2.waitKey(1)

        if len(good) > 0:
            q1 = np.float32([kp1[m.queryIdx].pt for m in good])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good])
            return q1, q2
        else:
            return np.array([]), np.array([])  # Return empty arrays if no good matches are found

    def get_pose(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        R, t = self.decomp_essential_mat(E, q1, q2)
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            T = self._form_transf(R, t)
            P1 = self.P
            P2 = np.matmul(self.P, T)
            hom_Q1 = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) /
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale
        return [R1, t]

    def update_plot(self):
        self.line.set_xdata(self.x_coords)
        self.line.set_ydata(self.z_coords)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_frame is not None:
                q1, q2 = self.get_matches(self.prev_frame, curr_frame)
                transf = self.get_pose(q1, q2)

                self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(transf))
                self.x_coords.append(self.cur_pose[0, 3])
                self.z_coords.append(self.cur_pose[2, 3])

                self.update_plot()

            self.prev_frame = curr_frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    vo = VisualOdometry()
    vo.run()