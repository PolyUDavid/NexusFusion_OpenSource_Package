"""
NexusFusion Basic Inference Example
==================================

This example demonstrates how to use the NexusFusion API for basic
multi-modal sensor fusion and trajectory prediction.

Authors: NexusFusion Research Team
License: MIT
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference import NexusFusionAPI


def generate_sample_sensor_data():
    """
    Generate sample sensor data for demonstration.
    
    In a real application, this data would come from actual sensors.
    
    Returns:
        dict: Sample multi-modal sensor data
    """
    # Sample LiDAR point cloud (1000 points)
    lidar_points = np.random.randn(1000, 4).astype(np.float32)
    lidar_points[:, :3] *= 10  # Scale spatial coordinates
    lidar_points[:, 3] = np.random.uniform(0, 1, 1000)  # Intensity values
    
    # Sample camera keypoints (500 keypoints)
    camera_keypoints = np.random.randn(500, 130).astype(np.float32)
    camera_keypoints[:, :2] *= 640  # Scale to image coordinates
    camera_keypoints[:, 2:] = np.random.randn(500, 128)  # Feature descriptors
    
    # Sample GNSS coordinates
    gnss = np.array([37.7749, -122.4194, 10.0], dtype=np.float32)  # San Francisco
    
    # Sample IMU sequence (20 timesteps)
    imu = np.random.randn(20, 6).astype(np.float32)
    imu[:, :3] *= 2  # Acceleration
    imu[:, 3:] *= 0.5  # Angular velocity
    
    # Sample V2X vehicle states (5 vehicles)
    v2x_states = np.random.randn(5, 8).astype(np.float32)
    v2x_states[:, :2] *= 100  # Position
    v2x_states[:, 2:4] *= 20   # Velocity
    v2x_states[:, 4] = np.random.uniform(0, 2*np.pi, 5)  # Heading
    
    return {
        'lidar_points': lidar_points,
        'camera_keypoints': camera_keypoints,
        'gnss': gnss,
        'imu': imu,
        'v2x_states': v2x_states
    }


def generate_peer_data(num_peers=3):
    """
    Generate sample peer vehicle data for cooperative inference.
    
    Args:
        num_peers (int): Number of peer vehicles
        
    Returns:
        list: List of peer sensor data dictionaries
    """
    peer_data = []
    for i in range(num_peers):
        peer_sensors = generate_sample_sensor_data()
        # Add some variation to peer data
        peer_sensors['gnss'] += np.random.randn(3) * 0.001  # Small position offset
        peer_data.append(peer_sensors)
    
    return peer_data


def main():
    """Main demonstration function."""
    print("🚀 NexusFusion Basic Inference Example")
    print("=" * 50)
    
    # Initialize NexusFusion API
    print("📦 Initializing NexusFusion API...")
    api = NexusFusionAPI(device='auto', precision='fp32')
    
    # Load models
    print("🔄 Loading models...")
    try:
        api.load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Generate sample data
    print("\n📊 Generating sample sensor data...")
    sensor_data = generate_sample_sensor_data()
    
    print(f"  - LiDAR points: {sensor_data['lidar_points'].shape}")
    print(f"  - Camera keypoints: {sensor_data['camera_keypoints'].shape}")
    print(f"  - GNSS coordinates: {sensor_data['gnss'].shape}")
    print(f"  - IMU sequence: {sensor_data['imu'].shape}")
    print(f"  - V2X states: {sensor_data['v2x_states'].shape}")
    
    # Single vehicle inference
    print("\n🚗 Running single vehicle inference...")
    start_time = time.time()
    
    try:
        results = api.predict(sensor_data)
        inference_time = time.time() - start_time
        
        print("✅ Inference completed successfully")
        print(f"  - Inference time: {inference_time*1000:.1f}ms")
        print(f"  - Trajectory shape: {results['trajectories'].shape}")
        print(f"  - Mode probabilities: {results['mode_probabilities']}")
        print(f"  - Collision risk: {results['collision_risk'][0]:.3f}")
        print(f"  - Intentions: {results['intentions']}")
        
    except Exception as e:
        print(f"❌ Single vehicle inference failed: {e}")
        return
    
    # Cooperative inference with peer data
    print("\n🚗🚗🚗 Running cooperative inference...")
    peer_data = generate_peer_data(num_peers=3)
    
    start_time = time.time()
    
    try:
        coop_results = api.predict(sensor_data, peer_data=peer_data)
        coop_inference_time = time.time() - start_time
        
        print("✅ Cooperative inference completed successfully")
        print(f"  - Inference time: {coop_inference_time*1000:.1f}ms")
        print(f"  - Number of peers: {coop_results['num_peers']}")
        print(f"  - Consensus confidence: {coop_results['consensus_confidence'][0]:.3f}")
        print(f"  - Trajectory shape: {coop_results['trajectories'].shape}")
        print(f"  - Collision risk: {coop_results['collision_risk'][0]:.3f}")
        
        # Compare single vs cooperative results
        risk_improvement = results['collision_risk'][0] - coop_results['collision_risk'][0]
        print(f"\n📈 Cooperative Benefits:")
        print(f"  - Risk reduction: {risk_improvement:.3f}")
        print(f"  - Consensus confidence: {coop_results['consensus_confidence'][0]:.3f}")
        
    except Exception as e:
        print(f"❌ Cooperative inference failed: {e}")
        return
    
    # Performance statistics
    print("\n📊 Performance Statistics:")
    stats = api.get_performance_stats()
    print(f"  - Total API calls: {stats['total_calls']}")
    print(f"  - Average latency: {stats['avg_latency']*1000:.1f}ms")
    print(f"  - Total processing time: {stats['total_time']:.2f}s")
    
    # Trajectory analysis
    print("\n🛣️ Trajectory Analysis:")
    trajectories = coop_results['trajectories'][0]  # First batch item
    mode_probs = coop_results['mode_probabilities'][0]
    
    # Find most likely trajectory
    best_mode = np.argmax(mode_probs)
    best_trajectory = trajectories[best_mode]
    
    print(f"  - Most likely mode: {best_mode} (confidence: {mode_probs[best_mode]:.3f})")
    print(f"  - Trajectory length: {len(best_trajectory)} timesteps")
    print(f"  - Final position: ({best_trajectory[-1, 0]:.2f}, {best_trajectory[-1, 1]:.2f})")
    
    # Compute trajectory statistics
    distances = np.linalg.norm(np.diff(best_trajectory, axis=0), axis=1)
    speeds = distances / 0.1  # Assuming 0.1s timestep
    
    print(f"  - Average speed: {np.mean(speeds):.2f} m/s")
    print(f"  - Max speed: {np.max(speeds):.2f} m/s")
    print(f"  - Total distance: {np.sum(distances):.2f} m")
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("  - Try with real sensor data")
    print("  - Experiment with different numbers of peers")
    print("  - Integrate with your autonomous driving stack")
    print("  - Check out the training example for custom models")


def demonstrate_model_components():
    """Demonstrate individual model components."""
    print("\n🔧 Model Components Demonstration")
    print("-" * 40)
    
    from models import create_mmf_gnn, create_sa_bft, create_ktf_transformer
    
    # Test MMF-GNN
    print("Testing MMF-GNN...")
    mmf_gnn = create_mmf_gnn()
    print(f"  - Parameters: {sum(p.numel() for p in mmf_gnn.parameters()):,}")
    
    # Test SA-BFT
    print("Testing SA-BFT...")
    sa_bft = create_sa_bft()
    print(f"  - Parameters: {sum(p.numel() for p in sa_bft.parameters()):,}")
    
    # Test KTF
    print("Testing KTF...")
    ktf = create_ktf_transformer()
    print(f"  - Parameters: {sum(p.numel() for p in ktf.parameters()):,}")
    
    total_params = (sum(p.numel() for p in mmf_gnn.parameters()) +
                   sum(p.numel() for p in sa_bft.parameters()) +
                   sum(p.numel() for p in ktf.parameters()))
    
    print(f"\n📊 Total NexusFusion Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
    demonstrate_model_components()
