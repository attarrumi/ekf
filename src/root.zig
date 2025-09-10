const std = @import("std");
const math = std.math;

pub const EKF = struct {
    // State vector (quaternion): [q0, q1, q2, q3]
    x: []f32,
    // State covariance matrix (4x4)
    P: []f32,
    // Process noise covariance (4x4)
    Q: []f32,
    // Measurement noise covariance (6x6)
    R: []f32,
    // Time step
    dt: f32,

    allocator: std.mem.Allocator,

    accel_filt: [3]f32,
    mag_filt: [3]f32,
    alpha_accel: f32,
    alpha_mag: f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, P: f32, Q: f32, R: f32) !Self {
        var ekf = Self{
            .x = try allocator.alloc(f32, 4),
            .P = try allocator.alloc(f32, 16), // 4x4
            .Q = try allocator.alloc(f32, 16), // 4x4
            .R = try allocator.alloc(f32, 36), // 6x6
            .dt = 0.01,
            .allocator = allocator,
            .accel_filt = .{ 0.0, 0.0, 0.0 },
            .mag_filt = .{ 0.0, 0.0, 0.0 },
            .alpha_accel = 0.33,
            .alpha_mag = 0.33,
        };

        // Initialize quaternion: [1, 0, 0, 0]
        ekf.x[0] = 1.0;
        ekf.x[1] = 0.0;
        ekf.x[2] = 0.0;
        ekf.x[3] = 0.0;

        // Initialize P as 0.1 * identity
        for (0..16) |i| {
            ekf.P[i] = if (i % 5 == 0) P else 0.0; // Diagonal elements
            // ekf.P[i] = if (i % 5 == 0) 0.01 else 0.0; // Diagonal elements
        }

        // Initialize Q as 0.0001 * identity
        for (0..16) |i| {
            ekf.Q[i] = if (i % 5 == 0) Q else 0.0;
            // ekf.Q[i] = if (i % 5 == 0) 0.0001 else 0.0;
        }

        // Initialize R as 0.01 * identity
        for (0..36) |i| {
            ekf.R[i] = if (i % 7 == 0) R else 0.0;
            // ekf.R[i] = if (i % 7 == 0) 0.01 else 0.0;
        }

        return ekf;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.x);
        self.allocator.free(self.P);
        self.allocator.free(self.Q);
        self.allocator.free(self.R);
    }

    // Quaternion multiplication matrix (right multiply)
    fn quatMultMatrix(q: [4]f32) [4][4]f32 {
        const q0 = q[0];
        const q1 = q[1];
        const q2 = q[2];
        const q3 = q[3];

        return .{
            .{ q0, -q1, -q2, -q3 },
            .{ q1, q0, -q3, q2 },
            .{ q2, q3, q0, -q1 },
            .{ q3, -q2, q1, q0 },
        };
    }

    // Normalize quaternion
    fn normalizeQuat(q: [4]f32) [4]f32 {
        var norm: f32 = 0.0;
        for (0..4) |i| {
            norm += q[i] * q[i];
        }
        norm = std.math.sqrt(norm);
        if (norm < 1e-10) return q; // Avoid division by zero
        return .{ q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm };
    }

    // Normalize quaternion in-place for slice
    fn normalizeQuatSlice(self: *Self) void {
        var norm: f32 = 0.0;
        for (0..4) |i| {
            norm += self.x[i] * self.x[i];
        }
        norm = std.math.sqrt(norm);
        if (norm < 1e-10) return; // Avoid division by zero
        for (0..4) |i| {
            self.x[i] /= norm;
        }
    }

    fn lowPassFilter(prev: *[3]f32, input: [3]f32, alpha: f32) void {
        for (0..3) |i| {
            prev[i] = alpha * input[i] + (1.0 - alpha) * prev[i];
        }
    }

    // Matrix multiplication: [m x n] * [n x p] -> [m x p]
    fn matMul(m: usize, n: usize, p: usize, a: []const f32, b: []const f32, out: []f32) void {
        std.debug.assert(a.len >= m * n);
        std.debug.assert(b.len >= n * p);
        std.debug.assert(out.len >= m * p);
        for (0..m) |i| {
            for (0..p) |j| {
                var sum: f32 = 0.0;
                for (0..n) |k| {
                    sum += a[i * n + k] * b[k * p + j];
                }
                out[i * p + j] = sum;
            }
        }
    }

    // Matrix transpose: [m x n] -> [n x m]
    fn matTranspose(m: usize, n: usize, a: []const f32, out: []f32) void {
        std.debug.assert(a.len >= m * n);
        std.debug.assert(out.len >= n * m);
        for (0..m) |i| {
            for (0..n) |j| {
                out[j * m + i] = a[i * n + j];
            }
        }
    }

    // Matrix addition: [m x n] + [m x n] -> [m x n]
    fn matAdd(m: usize, n: usize, a: []const f32, b: []const f32, out: []f32) void {
        std.debug.assert(a.len >= m * n);
        std.debug.assert(b.len >= m * n);
        std.debug.assert(out.len >= m * n);
        for (0..m * n) |i| {
            out[i] = a[i] + b[i];
        }
    }

    // Matrix subtraction: [m x n] - [m x n] -> [m x n]
    fn matSub(m: usize, n: usize, a: []const f32, b: []const f32, out: []f32) void {
        std.debug.assert(a.len >= m * n);
        std.debug.assert(b.len >= m * n);
        std.debug.assert(out.len >= m * n);
        for (0..m * n) |i| {
            out[i] = a[i] - b[i];
        }
    }

    // Matrix inverse (n x n) using Gaussian elimination with partial pivoting
    fn matInverse(n: usize, a: []const f32, out: []f32, allocator: std.mem.Allocator) !void {
        std.debug.assert(a.len >= n * n);
        std.debug.assert(out.len >= n * n);
        var aug = try allocator.alloc(f32, n * 2 * n);
        defer allocator.free(aug);

        // Initialize augmented matrix [A | I]
        for (0..n) |i| {
            for (0..n) |j| {
                aug[i * (2 * n) + j] = a[i * n + j];
                aug[i * (2 * n) + (j + n)] = if (i == j) 1.0 else 0.0;
            }
        }

        // Gaussian elimination with partial pivoting
        for (0..n) |col| {
            // Find pivot
            var pivot: usize = col;
            for (col + 1..n) |i| {
                if (@abs(aug[i * (2 * n) + col]) > @abs(aug[pivot * (2 * n) + col])) {
                    pivot = i;
                }
            }

            // Swap rows
            if (pivot != col) {
                for (0..2 * n) |j| {
                    const temp = aug[col * (2 * n) + j];
                    aug[col * (2 * n) + j] = aug[pivot * (2 * n) + j];
                    aug[pivot * (2 * n) + j] = temp;
                }
            }

            // Check for singularity
            if (@abs(aug[col * (2 * n) + col]) < 1e-10) {
                return error.SingularMatrix;
            }

            // Normalize pivot row
            const div = aug[col * (2 * n) + col];
            for (col..2 * n) |j| {
                aug[col * (2 * n) + j] /= div;
            }

            // Eliminate column
            for (0..n) |row| {
                if (row == col) continue;
                const factor = aug[row * (2 * n) + col];
                for (col..2 * n) |j| {
                    aug[row * (2 * n) + j] -= factor * aug[col * (2 * n) + j];
                }
            }
        }

        // Extract inverse
        for (0..n) |i| {
            for (0..n) |j| {
                out[i * n + j] = aug[i * (2 * n) + (j + n)];
            }
        }
    }

    fn applyMagDeclination(mag: [3]f32, declination: f32) [3]f32 {
        const cosD = std.math.cos(declination);
        const sinD = std.math.sin(declination);
        return .{
            mag[0] * cosD - mag[1] * sinD,
            mag[0] * sinD + mag[1] * cosD,
            mag[2],
        };
    }

    // Predict step using gyroscope data
    pub fn predict(self: *Self, gyro: [3]f32, dt: f32) void {
        // Angular velocities
        const wx = gyro[0];
        const wy = gyro[1];
        const wz = gyro[2];

        self.dt = dt;

        // Construct omega matrix (4x4)
        var omega = [_]f32{0} ** 16;
        omega[0 * 4 + 1] = -wx * 0.5 * self.dt;
        omega[0 * 4 + 2] = -wy * 0.5 * self.dt;
        omega[0 * 4 + 3] = -wz * 0.5 * self.dt;
        omega[1 * 4 + 0] = wx * 0.5 * self.dt;
        omega[1 * 4 + 2] = wz * 0.5 * self.dt;
        omega[1 * 4 + 3] = -wy * 0.5 * self.dt;
        omega[2 * 4 + 0] = wy * 0.5 * self.dt;
        omega[2 * 4 + 1] = -wz * 0.5 * self.dt;
        omega[2 * 4 + 3] = wx * 0.5 * self.dt;
        omega[3 * 4 + 0] = wz * 0.5 * self.dt;
        omega[3 * 4 + 1] = wy * 0.5 * self.dt;
        omega[3 * 4 + 2] = -wx * 0.5 * self.dt;

        // State transition matrix F = I + omega
        var F = [_]f32{0} ** 16;
        for (0..4) |i| {
            F[i * 4 + i] = 1.0;
        }
        matAdd(4, 4, &F, &omega, &F);

        // Predict state: x = F * x
        var new_x = [_]f32{0} ** 4;
        matMul(4, 4, 1, &F, self.x, &new_x);
        for (0..4) |i| {
            self.x[i] = new_x[i];
        }
        self.normalizeQuatSlice();

        // Predict covariance: P = F * P * F' + Q
        var temp = [_]f32{0} ** 16;
        var Ft = [_]f32{0} ** 16;
        matTranspose(4, 4, &F, &Ft);
        matMul(4, 4, 4, &F, self.P, &temp);
        matMul(4, 4, 4, &temp, &Ft, &temp);
        matAdd(4, 4, &temp, self.Q, self.P);
    }

    // Update step using accelerometer and magnetometer
    pub fn update(self: *Self, accel: [3]f32, mag: [3]f32) !void {
        self.accel_filt = accel;
        self.mag_filt = mag;
        // Filter sensor dulu
        // lowPassFilter(&self.accel_filt, accel, self.alpha_accel);
        // lowPassFilter(&self.mag_filt, mag, self.alpha_mag);

        // Pakai hasil filter
        var accel_norm = self.accel_filt;
        var mag_norm = self.mag_filt;

        mag_norm = applyMagDeclination(mag_norm, 0.0133808576);

        var norm: f32 = 0.0;
        for (0..3) |i| {
            norm += accel[i] * accel[i];
        }
        norm = std.math.sqrt(norm);
        if (norm > 1e-10) {
            for (0..3) |i| {
                accel_norm[i] /= norm;
            }
        }

        norm = 0.0;
        for (0..3) |i| {
            norm += mag[i] * mag[i];
        }
        norm = std.math.sqrt(norm);
        if (norm > 1e-10) {
            for (0..3) |i| {
                mag_norm[i] /= norm;
            }
        }

        // Current quaternion
        const q0 = self.x[0];
        const q1 = self.x[1];
        const q2 = self.x[2];
        const q3 = self.x[3];

        // Expected measurements h(x)
        var h = [_]f32{
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 + q0 * q3),
            2.0 * (q2 * q3 - q0 * q1),
        };

        // Measurement vector z
        var z = [_]f32{
            accel_norm[0],
            accel_norm[1],
            accel_norm[2],
            mag_norm[0],
            mag_norm[1],
            mag_norm[2],
        };

        // Jacobian matrix H (6x4)
        var H = [_]f32{0} ** 24;
        H[0 * 4 + 0] = -2.0 * q2;
        H[0 * 4 + 1] = 2.0 * q3;
        H[0 * 4 + 2] = -2.0 * q0;
        H[0 * 4 + 3] = 2.0 * q1;
        H[1 * 4 + 0] = 2.0 * q1;
        H[1 * 4 + 1] = 2.0 * q0;
        H[1 * 4 + 2] = 2.0 * q3;
        H[1 * 4 + 3] = 2.0 * q2;
        H[2 * 4 + 0] = 2.0 * q0;
        H[2 * 4 + 1] = -2.0 * q1;
        H[2 * 4 + 2] = -2.0 * q2;
        H[2 * 4 + 3] = 2.0 * q3;
        H[3 * 4 + 0] = 2.0 * q0;
        H[3 * 4 + 1] = 2.0 * q1;
        H[3 * 4 + 2] = -2.0 * q2;
        H[3 * 4 + 3] = -2.0 * q3;
        H[4 * 4 + 0] = 2.0 * q3;
        H[4 * 4 + 1] = 2.0 * q2;
        H[4 * 4 + 2] = 2.0 * q1;
        H[4 * 4 + 3] = 2.0 * q0;
        H[5 * 4 + 0] = -2.0 * q1;
        H[5 * 4 + 1] = -2.0 * q0;
        H[5 * 4 + 2] = 2.0 * q3;
        H[5 * 4 + 3] = 2.0 * q2;

        // Innovation: y = z - h
        var y = [_]f32{0} ** 6;
        matSub(6, 1, &z, &h, &y);

        // Innovation covariance: S = H * P * H' + R
        var Ht = [_]f32{0} ** 24; // 4x6
        var temp = [_]f32{0} ** 24; // 6x4
        var S = [_]f32{0} ** 36; // 6x6
        matTranspose(6, 4, &H, &Ht);
        matMul(6, 4, 4, &H, self.P, &temp);
        matMul(6, 4, 6, &temp, &Ht, &S);
        matAdd(6, 6, &S, self.R, &S);

        // Inverse of S
        var inv_S = [_]f32{0} ** 36; // 6x6
        try matInverse(6, &S, &inv_S, self.allocator);
        // Kalman gain: K = P * H' * inv(S)
        var temp2 = [_]f32{0} ** 24; // 4x6
        var K = [_]f32{0} ** 24; // 4x6
        matMul(4, 4, 6, self.P, &Ht, &temp2);
        matMul(4, 6, 6, &temp2, &inv_S, &K);

        // Update state: x = x + K * y
        var K_y = [_]f32{0} ** 4;
        matMul(4, 6, 1, &K, &y, &K_y);
        for (0..4) |i| {
            self.x[i] += K_y[i];
        }
        self.normalizeQuatSlice();

        // Update covariance: P = (I - K * H) * P
        var KH = [_]f32{0} ** 16; // 4x4
        var I_minus_KH = [_]f32{0} ** 16; // 4x4
        var new_P = [_]f32{0} ** 16; // 4x4
        matMul(4, 6, 4, &K, &H, &KH);
        for (0..4) |i| {
            I_minus_KH[i * 4 + i] = 1.0;
        }
        matSub(4, 4, &I_minus_KH, &KH, &I_minus_KH);
        matMul(4, 4, 4, &I_minus_KH, self.P, &new_P);
        @memcpy(self.P, &new_P);
    }

    // Get current quaternion
    pub fn getQuaternion(self: Self) [4]f32 {
        return .{ self.x[0], self.x[1], self.x[2], self.x[3] };
    }

    pub fn quaternionToEuler(self: Self) [3]f32 {
        const q0 = self.x[0];
        const q1 = self.x[1];
        const q2 = self.x[2];
        const q3 = self.x[3];

        const roll = std.math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2));
        const pitch = std.math.asin(2.0 * (q0 * q2 - q3 * q1));
        const yaw = std.math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3));
        return .{ roll, pitch, yaw };
    }
};
