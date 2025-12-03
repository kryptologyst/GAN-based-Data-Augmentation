"""Streamlit demo for GAN-based data augmentation."""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.dcgan import DCGAN
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.visualization import save_image_grid


@st.cache_resource
def load_model(checkpoint_path: str, config: dict, device: torch.device) -> DCGAN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    else:
        model_state_dict = checkpoint
    
    model = DCGAN(
        generator_config=config["model"]["generator"],
        discriminator_config=config["model"]["discriminator"],
    )
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model


def generate_samples(model: DCGAN, num_samples: int, device: torch.device, seed: int) -> torch.Tensor:
    """Generate samples with fixed seed."""
    set_seed(seed)
    with torch.no_grad():
        z = torch.randn(num_samples, model.generator.z_dim, device=device)
        samples = model.generator(z)
    return samples


def tensor_to_plotly_image(tensor: torch.Tensor, title: str = "") -> go.Figure:
    """Convert tensor to plotly image."""
    # Convert to numpy and denormalize
    img = tensor.cpu().numpy()
    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
    img = np.clip(img, 0, 1)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Image(z=img[0, 0] if img.shape[1] == 1 else img[0].transpose(1, 2, 0)))
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GAN Data Augmentation Demo",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 GAN-based Data Augmentation Demo")
    st.markdown("Generate synthetic images using a trained DCGAN model for data augmentation.")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Model selection
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            checkpoint_path = st.sidebar.selectbox(
                "Select Model Checkpoint",
                checkpoints,
                format_func=lambda x: x.name,
            )
        else:
            st.error("No model checkpoints found. Please train a model first.")
            return
    else:
        st.error("Checkpoints directory not found. Please train a model first.")
        return
    
    # Load configuration
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/config.yaml")
        config = OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return
    
    # Device selection
    device = get_device(config.get("device", "auto"))
    
    # Load model
    try:
        model = load_model(str(checkpoint_path), config, device)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    # Generation controls
    st.sidebar.subheader("Generation Settings")
    
    num_samples = st.sidebar.slider("Number of Samples", 1, 64, 16)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=2**32-1)
    
    # Generate button
    if st.sidebar.button("Generate Samples", type="primary"):
        with st.spinner("Generating samples..."):
            samples = generate_samples(model, num_samples, device, seed)
        
        st.success(f"Generated {num_samples} samples!")
        
        # Display samples in a grid
        cols = st.columns(4)
        for i, sample in enumerate(samples):
            col_idx = i % 4
            with cols[col_idx]:
                fig = tensor_to_plotly_image(sample, f"Sample {i+1}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Interpolation section
    st.sidebar.subheader("Latent Interpolation")
    
    if st.sidebar.button("Generate Interpolation"):
        with st.spinner("Generating interpolation..."):
            set_seed(seed)
            
            # Generate two random noise vectors
            z1 = torch.randn(1, model.generator.z_dim, device=device)
            z2 = torch.randn(1, model.generator.z_dim, device=device)
            
            # Interpolate
            steps = 10
            alphas = torch.linspace(0, 1, steps, device=device)
            interpolated_samples = []
            
            with torch.no_grad():
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    sample = model.generator(z_interp)
                    interpolated_samples.append(sample)
            
            interpolated_samples = torch.cat(interpolated_samples, dim=0)
        
        st.success("Generated interpolation!")
        
        # Display interpolation
        st.subheader("Latent Space Interpolation")
        cols = st.columns(steps)
        for i, sample in enumerate(interpolated_samples):
            with cols[i]:
                fig = tensor_to_plotly_image(sample, f"Step {i+1}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Latent traversal section
    st.sidebar.subheader("Latent Traversal")
    
    dim_idx = st.sidebar.slider("Dimension to Traverse", 0, model.generator.z_dim-1, 0)
    traversal_steps = st.sidebar.slider("Traversal Steps", 5, 20, 10)
    
    if st.sidebar.button("Generate Traversal"):
        with st.spinner("Generating latent traversal..."):
            set_seed(seed)
            
            # Generate base noise vector
            z_base = torch.randn(1, model.generator.z_dim, device=device)
            
            # Traverse along selected dimension
            range_vals = (-3, 3)
            traversal_samples = []
            
            with torch.no_grad():
                for val in np.linspace(range_vals[0], range_vals[1], traversal_steps):
                    z_traversal = z_base.clone()
                    z_traversal[0, dim_idx] = val
                    sample = model.generator(z_traversal)
                    traversal_samples.append(sample)
            
            traversal_samples = torch.cat(traversal_samples, dim=0)
        
        st.success(f"Generated traversal for dimension {dim_idx}!")
        
        # Display traversal
        st.subheader(f"Latent Space Traversal - Dimension {dim_idx}")
        cols = st.columns(traversal_steps)
        for i, sample in enumerate(traversal_samples):
            with cols[i]:
                fig = tensor_to_plotly_image(sample, f"Val {i+1}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"""
    **Model:** DCGAN
    **Latent Dim:** {model.generator.z_dim}
    **Image Size:** {model.generator.image_size}x{model.generator.image_size}
    **Channels:** {model.generator.channels}
    **Device:** {device}
    """)
    
    # About section
    st.sidebar.subheader("About")
    st.sidebar.markdown("""
    This demo showcases a DCGAN trained on MNIST for data augmentation.
    
    **Features:**
    - Generate random samples
    - Latent space interpolation
    - Latent dimension traversal
    
    **Use Cases:**
    - Data augmentation for small datasets
    - Understanding latent representations
    - Exploring generative model capabilities
    """)


if __name__ == "__main__":
    main()
