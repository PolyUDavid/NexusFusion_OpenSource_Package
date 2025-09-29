#!/usr/bin/env python3
"""
NexusFusion Data Augmentation Configuration
数据增强配置文件
"""

# 数据增强配置
AUGMENTATION_CONFIG = {
    # 全局设置
    "enabled": True,
    "probability": 0.5,
    "random_seed": 42,
    
    # LiDAR点云增强
    "lidar_augmentation": {
        "enabled": True,
        "techniques": {
            "gaussian_noise": {
                "enabled": True,
                "noise_scale": 0.01,
                "description": "高斯噪声注入，模拟传感器噪声"
            },
            "rotation": {
                "enabled": False,  # 暂时禁用以避免维度错误
                "range_degrees": 15,
                "description": "随机旋转±15度"
            },
            "translation": {
                "enabled": False,  # 暂时禁用
                "range_meters": 0.5,
                "description": "随机平移±0.5米"
            },
            "occlusion": {
                "enabled": False,  # 暂时禁用
                "ratio": 0.1,
                "description": "随机遮挡10%点云"
            }
        }
    },
    
    # GNSS增强
    "gnss_augmentation": {
        "enabled": True,
        "noise_scale": 0.1,  # 10cm噪声
        "description": "GPS定位噪声，模拟真实定位误差"
    },
    
    # IMU增强
    "imu_augmentation": {
        "enabled": True,
        "noise_scale": 0.05,  # 5%噪声
        "description": "IMU传感器噪声，模拟加速度计和陀螺仪误差"
    },
    
    # 相机关键点增强
    "camera_augmentation": {
        "enabled": False,  # 暂时禁用以避免维度错误
        "techniques": {
            "gaussian_noise": {
                "enabled": True,
                "noise_scale": 0.02
            },
            "dropout": {
                "enabled": True,
                "ratio": 0.05
            }
        }
    },
    
    # 轨迹增强
    "trajectory_augmentation": {
        "enabled": False,  # 暂时禁用
        "noise_scale": 0.01,
        "description": "轨迹点噪声"
    }
}

# 增强策略说明
AUGMENTATION_STRATEGIES = {
    "conservative": {
        "description": "保守策略，仅使用基础噪声增强",
        "probability": 0.3,
        "enabled_techniques": ["lidar_gaussian_noise", "gnss_noise", "imu_noise"]
    },
    "moderate": {
        "description": "中等策略，当前使用的配置",
        "probability": 0.5,
        "enabled_techniques": ["lidar_gaussian_noise", "gnss_noise", "imu_noise"]
    },
    "aggressive": {
        "description": "激进策略，使用所有增强技术（需要调试）",
        "probability": 0.7,
        "enabled_techniques": ["all"]
    }
}

# 防过拟合配置
ANTI_OVERFITTING_CONFIG = {
    "regularization": {
        "dropout": 0.2,
        "weight_decay": 3e-4,
        "gradient_clipping": 1.0
    },
    "learning_rate": {
        "peak_reduction": 0.5,  # 峰值降到50%
        "warmup_ratio": 0.12,
        "stable_ratio": 0.25,
        "decay_ratio": 0.63
    },
    "early_stopping": {
        "patience": 10,
        "min_delta": 0.001
    },
    "gradient_accumulation": {
        "steps": 4,
        "effective_batch_size": 16  # 4 * 4
    }
}

if __name__ == "__main__":
    import json
    print("=== NexusFusion 数据增强配置 ===")
    print(json.dumps(AUGMENTATION_CONFIG, indent=2, ensure_ascii=False))
    print("\n=== 防过拟合配置 ===")
    print(json.dumps(ANTI_OVERFITTING_CONFIG, indent=2, ensure_ascii=False))
