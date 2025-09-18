const std = @import("std");
const math = std.math;

pub fn EKF(comptime T: type) type {
    return struct {
        const Self = @This();
        x: [7]T,
        P: [7][7]T,
        Q: [7][7]T,
        R_acc: [3][3]T,
        R_mag: [3][3]T,

        // GANTI: Tambahkan 4 parameter ke signature fungsi
        pub fn init(q_quat: f64, q_bias: f64, r_acc: f64, r_mag: f64) Self {
            var ekf = Self{
                .x = .{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                .P = std.mem.zeroes([7][7]f64),
                .Q = std.mem.zeroes([7][7]f64),
                .R_acc = std.mem.zeroes([3][3]f64),
                .R_mag = std.mem.zeroes([3][3]f64),
            };

            // Inisialisasi P tetap sama
            for (0..7) |i| {
                ekf.P[i][i] = if (i < 4) 0.1 else 0.01;
            }

            // Process noise - GANTI: Gunakan parameter dari argumen
            for (0..4) |i| {
                ekf.Q[i][i] = q_quat; // Menggunakan q_quat
            }
            for (4..7) |i| {
                ekf.Q[i][i] = q_bias; // Menggunakan q_bias
            }

            // Measurement noise - GANTI: Gunakan parameter dari argumen
            for (0..3) |i| {
                ekf.R_acc[i][i] = r_acc; // Menggunakan r_acc
                ekf.R_mag[i][i] = r_mag; // Menggunakan r_mag
            }

            return ekf;
        }

        pub fn predict(self: *Self, gx: T, gy: T, gz: T, dt: T) void {
            // Remove bias from gyroscope measurements
            const wx = gx - self.x[4];
            const wy = gy - self.x[5];
            const wz = gz - self.x[6];

            // Current quaternion
            const q0 = self.x[0];
            const q1 = self.x[1];
            const q2 = self.x[2];
            const q3 = self.x[3];

            // Quaternion derivative matrix (omega matrix)
            const omega = [_][4]T{
                .{ 0.0, -wx, -wy, -wz },
                .{ wx, 0.0, wz, -wy },
                .{ wy, -wz, 0.0, wx },
                .{ wz, wy, -wx, 0.0 },
            };

            // Apply quaternion update: q_dot = 0.5 * omega * q
            const half_dt = 0.5 * dt;
            const q_current = [4]T{ q0, q1, q2, q3 };

            // Matrix-vector multiplication: omega * q
            var q_dot = [4]T{ 0.0, 0.0, 0.0, 0.0 };
            for (0..4) |i| {
                for (0..4) |j| {
                    q_dot[i] += omega[i][j] * q_current[j];
                }
            }

            // Integrate: q_new = q + half_dt * q_dot
            self.x[0] = q0 + half_dt * q_dot[0];
            self.x[1] = q1 + half_dt * q_dot[1];
            self.x[2] = q2 + half_dt * q_dot[2];
            self.x[3] = q3 + half_dt * q_dot[3];

            // Normalize quaternion
            self.normalizeQuaternion();

            // Jacobian matrix F
            var F = std.mem.zeroes([7][7]T);

            // Identity for bias states
            F[4][4] = 1.0;
            F[5][5] = 1.0;
            F[6][6] = 1.0;

            // Quaternion Jacobian
            F[0][0] = 1.0;
            F[0][1] = -half_dt * wx;
            F[0][2] = -half_dt * wy;
            F[0][3] = -half_dt * wz;
            F[0][4] = half_dt * q1;
            F[0][5] = half_dt * q2;
            F[0][6] = half_dt * q3;

            F[1][0] = half_dt * wx;
            F[1][1] = 1.0;
            F[1][2] = half_dt * wz;
            F[1][3] = -half_dt * wy;
            F[1][4] = -half_dt * q0;
            F[1][5] = half_dt * q3;
            F[1][6] = -half_dt * q2;

            F[2][0] = half_dt * wy;
            F[2][1] = -half_dt * wz;
            F[2][2] = 1.0;
            F[2][3] = half_dt * wx;
            F[2][4] = -half_dt * q3;
            F[2][5] = -half_dt * q0;
            F[2][6] = half_dt * q1;

            F[3][0] = half_dt * wz;
            F[3][1] = half_dt * wy;
            F[3][2] = -half_dt * wx;
            F[3][3] = 1.0;
            F[3][4] = half_dt * q2;
            F[3][5] = -half_dt * q1;
            F[3][6] = -half_dt * q0;

            // Predict covariance: P = F*P*F^T + Q
            var FP = std.mem.zeroes([7][7]T);
            matrixMultiply(F, self.P, &FP);

            var FPFt = std.mem.zeroes([7][7]T);
            matrixMultiplyTranspose(FP, F, &FPFt);

            matrixAdd(FPFt, self.Q, &self.P);
        }

        pub fn updateAccelerometer(self: *Self, ax: T, ay: T, az: T) void {
            // Normalize accelerometer
            const norm = math.sqrt(ax * ax + ay * ay + az * az);
            if (norm < 1e-6) return;

            const ax_norm = ax / norm;
            const ay_norm = ay / norm;
            const az_norm = az / norm;

            // Expected gravity vector in body frame
            const q0 = self.x[0];
            const q1 = self.x[1];
            const q2 = self.x[2];
            const q3 = self.x[3];

            const expected = [3]T{
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q0 * q1 + q2 * q3),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            };

            // Measurement residual
            const residual = [3]T{
                ax_norm - expected[0],
                ay_norm - expected[1],
                az_norm - expected[2],
            };

            // Measurement Jacobian H
            var H = std.mem.zeroes([3][7]T);
            H[0][0] = -2.0 * q2;
            H[0][1] = 2.0 * q3;
            H[0][2] = -2.0 * q0;
            H[0][3] = 2.0 * q1;

            H[1][0] = 2.0 * q1;
            H[1][1] = 2.0 * q0;
            H[1][2] = 2.0 * q3;
            H[1][3] = 2.0 * q2;

            H[2][0] = 2.0 * q0;
            H[2][1] = -2.0 * q1;
            H[2][2] = -2.0 * q2;
            H[2][3] = 2.0 * q3;

            self.kalmanUpdate(H, residual, self.R_acc);
        }

        pub fn updateMagnetometer(self: *Self, mx: T, my: T, mz: T) void {
            // Normalize magnetometer
            const norm = math.sqrt(mx * mx + my * my + mz * mz);
            if (norm < 1e-6) return;

            const mx_norm = mx / norm;
            const my_norm = my / norm;
            const mz_norm = mz / norm;

            const q0 = self.x[0];
            const q1 = self.x[1];
            const q2 = self.x[2];
            const q3 = self.x[3];

            // Rotate magnetometer to earth frame to find reference
            const hx = 2.0 * mx_norm * (0.5 - q2 * q2 - q3 * q3) +
                2.0 * my_norm * (q1 * q2 - q0 * q3) +
                2.0 * mz_norm * (q1 * q3 + q0 * q2);
            const hy = 2.0 * mx_norm * (q1 * q2 + q0 * q3) +
                2.0 * my_norm * (0.5 - q1 * q1 - q3 * q3) +
                2.0 * mz_norm * (q2 * q3 - q0 * q1);

            const bx = math.sqrt(hx * hx + hy * hy);
            const bz = 2.0 * mx_norm * (q1 * q3 - q0 * q2) +
                2.0 * my_norm * (q2 * q3 + q0 * q1) +
                2.0 * mz_norm * (0.5 - q1 * q1 - q2 * q2);

            // Expected magnetic field in body frame
            const expected = [3]T{
                2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2),
                2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3),
                2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2),
            };

            const residual = [3]T{
                mx_norm - expected[0],
                my_norm - expected[1],
                mz_norm - expected[2],
            };

            // Simplified magnetometer Jacobian (approximate)
            var H = std.mem.zeroes([3][7]T);
            // This is a simplified version - full Jacobian is quite complex
            H[0][0] = -2.0 * bz * q2;
            H[0][1] = 2.0 * bz * q3;
            H[0][2] = -4.0 * bx * q2 - 2.0 * bz * q0;
            H[0][3] = -4.0 * bx * q3 + 2.0 * bz * q1;

            self.kalmanUpdate(H, residual, self.R_mag);
        }

        fn kalmanUpdate(self: *Self, H: [3][7]T, residual: [3]T, R: [3][3]T) void {
            // S = H*P*H^T + R
            var HP = std.mem.zeroes([3][7]T);
            matrixMultiply3x7(H, self.P, &HP);

            var S = R;
            for (0..3) |i| {
                for (0..3) |j| {
                    var sum: T = 0.0;
                    for (0..7) |k| {
                        sum += HP[i][k] * H[j][k];
                    }
                    S[i][j] += sum;
                }
            }

            // Kalman gain K = P*H^T*S^(-1)
            var Ht = std.mem.zeroes([7][3]T);
            transpose3x7(H, &Ht);

            var PHt = std.mem.zeroes([7][3]T);
            matrixMultiply7x3(self.P, Ht, &PHt);

            var S_inv = std.mem.zeroes([3][3]T);
            if (!matrixInvert3x3(S, &S_inv)) return; // Skip update if singular

            var K = std.mem.zeroes([7][3]T);
            matrixMultiply7x3_3x3(PHt, S_inv, &K);

            // Update state: x = x + K*residual
            for (0..7) |i| {
                for (0..3) |j| {
                    self.x[i] += K[i][j] * residual[j];
                }
            }

            // Update covariance: P = (I - K*H)*P
            var KH = std.mem.zeroes([7][7]T);
            matrixMultiply7x3_3x7(K, H, &KH);

            var I_KH = std.mem.zeroes([7][7]T);
            for (0..7) |i| {
                for (0..7) |j| {
                    I_KH[i][j] = if (i == j) 1.0 else 0.0;
                    I_KH[i][j] -= KH[i][j];
                }
            }

            var new_P = std.mem.zeroes([7][7]T);
            matrixMultiply(I_KH, self.P, &new_P);
            self.P = new_P;

            self.normalizeQuaternion();
        }

        pub fn update(self: *Self, gx: T, gy: T, gz: T, ax: T, ay: T, az: T, mx: T, my: T, mz: T, dt: T) void {
            self.predict(gx, gy, gz, dt);
            self.updateAccelerometer(ax, ay, az);
            self.updateMagnetometer(mx, my, mz);
        }

        pub fn getEuler(self: *const Self) struct { roll: T, pitch: T, yaw: T } {
            const q0 = self.x[0];
            const q1 = self.x[1];
            const q2 = self.x[2];
            const q3 = self.x[3];

            const roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
            const pitch = math.asin(2.0 * (q0 * q2 - q1 * q3));
            var yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);

            const rad_to_deg = 180.0 / math.pi;
            yaw -= 0.0133808576; // Your declination correction

            if (yaw > math.pi) yaw -= 2 * math.pi;
            if (yaw < -math.pi) yaw += 2 * math.pi;

            return .{
                .roll = roll * rad_to_deg,
                .pitch = pitch * rad_to_deg,
                .yaw = yaw * rad_to_deg,
            };
        }

        pub fn getQuaternion(self: *const Self) [4]T {
            return .{ self.x[0], self.x[1], self.x[2], self.x[3] };
        }

        fn normalizeQuaternion(self: *Self) void {
            const norm = math.sqrt(self.x[0] * self.x[0] + self.x[1] * self.x[1] +
                self.x[2] * self.x[2] + self.x[3] * self.x[3]);
            if (norm > 1e-6) {
                self.x[0] /= norm;
                self.x[1] /= norm;
                self.x[2] /= norm;
                self.x[3] /= norm;
            }
        }

        // Matrix operation helper functions
        fn matrixMultiply(A: [7][7]T, B: [7][7]T, C: *[7][7]T) void {
            for (0..7) |i| {
                for (0..7) |j| {
                    C[i][j] = 0.0;
                    for (0..7) |k| {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        fn matrixMultiplyTranspose(A: [7][7]T, B: [7][7]T, C: *[7][7]T) void {
            for (0..7) |i| {
                for (0..7) |j| {
                    C[i][j] = 0.0;
                    for (0..7) |k| {
                        C[i][j] += A[i][k] * B[j][k]; // B transposed
                    }
                }
            }
        }

        fn matrixAdd(A: [7][7]T, B: [7][7]T, C: *[7][7]T) void {
            for (0..7) |i| {
                for (0..7) |j| {
                    C[i][j] = A[i][j] + B[i][j];
                }
            }
        }

        fn matrixMultiply3x7(A: [3][7]T, B: [7][7]T, C: *[3][7]T) void {
            for (0..3) |i| {
                for (0..7) |j| {
                    C[i][j] = 0.0;
                    for (0..7) |k| {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        fn matrixMultiply7x3(A: [7][7]T, B: [7][3]T, C: *[7][3]T) void {
            for (0..7) |i| {
                for (0..3) |j| {
                    C[i][j] = 0.0;
                    for (0..7) |k| {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        fn matrixMultiply7x3_3x3(A: [7][3]T, B: [3][3]T, C: *[7][3]T) void {
            for (0..7) |i| {
                for (0..3) |j| {
                    C[i][j] = 0.0;
                    for (0..3) |k| {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        fn matrixMultiply7x3_3x7(A: [7][3]T, B: [3][7]T, C: *[7][7]T) void {
            for (0..7) |i| {
                for (0..7) |j| {
                    C[i][j] = 0.0;
                    for (0..3) |k| {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        fn transpose3x7(A: [3][7]T, At: *[7][3]T) void {
            for (0..3) |i| {
                for (0..7) |j| {
                    At[j][i] = A[i][j];
                }
            }
        }

        fn matrixInvert3x3(A: [3][3]T, A_inv: *[3][3]T) bool {
            const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

            if (@abs(det) < 1e-12) return false;

            const inv_det = 1.0 / det;

            A_inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
            A_inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
            A_inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
            A_inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
            A_inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
            A_inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;
            A_inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;
            A_inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;
            A_inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;

            return true;
        }
    };
}
