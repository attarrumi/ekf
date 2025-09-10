const ekf = b.dependency("ekf", .{ .target = target, .optimize = optimize, });

exe.root_module.addImport("ekf", matzig.module("ekf"));