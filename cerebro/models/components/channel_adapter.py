"""Channel adapter for handling variable input channels via Perceiver-style cross-attention.

This module enables training on mixed datasets with different channel counts (e.g., TUH 21ch + HBN 128ch)
by mapping all inputs to a unified channel representation.
"""

import torch
import torch.nn as nn


class PerceiverChannelAdapter(nn.Module):
    """Perceiver-style channel adapter using cross-attention.

    Maps variable input channels to fixed target channels using learnable queries.
    Enables mixing datasets with different channel counts (e.g., TUH 21ch, HBN 128ch).

    Architecture:
        1. Learnable query tokens (target_channels x d_model)
        2. Cross-attention: queries attend to input channels
        3. Output: (batch, target_channels, time) regardless of input channels

    Args:
        target_channels: Target number of channels (e.g., 128 for HBN)
        d_model: Embedding dimension for attention (default: 64)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> adapter = PerceiverChannelAdapter(target_channels=128)
        >>> tuh_input = torch.randn(32, 21, 200)   # TUH: 21 channels
        >>> hbn_input = torch.randn(32, 128, 200)  # HBN: 128 channels
        >>> tuh_out = adapter(tuh_input)  # (32, 128, 200)
        >>> hbn_out = adapter(hbn_input)  # (32, 128, 200)
    """

    def __init__(
        self,
        target_channels: int = 128,
        d_model: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_channels = target_channels
        self.d_model = d_model

        # Learnable query tokens (one per target channel)
        self.queries = nn.Parameter(torch.randn(target_channels, d_model))

        # Project input channels to d_model
        # Note: This will be applied per-sample since input channels vary
        # We use a lazy linear layer that adapts to input
        self.input_proj = None  # Initialized on first forward pass

        # Cross-attention: queries attend to input
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # (seq, batch, embed)
        )

        # Output projection: d_model → 1 (channel representation)
        self.output_proj = nn.Linear(d_model, 1)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with variable input channels.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Output tensor (batch, target_channels, time)
        """
        batch_size, in_channels, time_steps = x.shape

        # Initialize input projection if needed (lazy initialization)
        if self.input_proj is None or self.input_proj.in_features != in_channels:
            self.input_proj = nn.Linear(in_channels, self.d_model).to(x.device)

        # Reshape: (batch, channels, time) → (batch, time, channels)
        x = x.permute(0, 2, 1)  # (batch, time, in_channels)

        # Process each time step independently
        # Shape: (batch, time, in_channels) → (batch, time, d_model)
        x_proj = self.input_proj(x)  # (batch, time, d_model)

        # Prepare for cross-attention
        # Keys/Values: input channels (batch, time, d_model)
        # Queries: learnable tokens (target_channels, d_model)

        # Expand queries for batch: (target_channels, d_model) → (batch, target_channels, d_model)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Reshape for attention (seq, batch, embed) format
        # Queries: (target_channels, batch, d_model)
        # Keys/Values: (time, batch, d_model)
        queries_seq = queries.permute(1, 0, 2)  # (target_channels, batch, d_model)
        kv_seq = x_proj.permute(1, 0, 2)  # (time, batch, d_model)

        # Cross-attention: queries attend to input
        attn_out, _ = self.cross_attention(
            query=queries_seq,
            key=kv_seq,
            value=kv_seq,
        )  # (target_channels, batch, d_model)

        # Reshape back: (target_channels, batch, d_model) → (batch, target_channels, d_model)
        attn_out = attn_out.permute(1, 0, 2)

        # Layer norm + residual
        attn_out = self.norm(attn_out + queries)

        # For each target channel, aggregate across time via another attention pass
        # This produces the final (batch, target_channels, time) output

        # Alternative simpler approach: Project directly to time dimension
        # attn_out: (batch, target_channels, d_model)
        # We want: (batch, target_channels, time)

        # Use linear projection to map d_model → time_steps for each channel
        # This is computationally expensive, so let's use a different approach:

        # Better approach: Use attn_out as channel embeddings and broadcast to time
        # Then use another attention mechanism or convolution to spread across time

        # Simplest approach: Learn time-invariant channel mapping
        # attn_out: (batch, target_channels, d_model) → (batch, target_channels, 1)
        # Then broadcast to time

        # Actually, let's reconsider the architecture:
        # We want to map (batch, in_channels, time) → (batch, target_channels, time)
        # Perceiver typically works on flattened sequences, not 2D grids

        # Revised approach: Treat each time step independently
        output_time = []
        for t in range(time_steps):
            # Extract features at time t: (batch, in_channels)
            x_t = x[:, t, :]  # (batch, in_channels)

            # Project to d_model: (batch, in_channels) → (batch, d_model)
            x_t_proj = self.input_proj(x_t)  # (batch, d_model)

            # Cross-attention with queries
            # Queries: (target_channels, d_model)
            # Keys/Values: (batch, d_model)

            # Expand for batch: (target_channels, batch, d_model)
            q = self.queries.unsqueeze(1).expand(-1, batch_size, -1)

            # Keys/Values: (1, batch, d_model)
            k = v = x_t_proj.unsqueeze(0)

            # Attention
            attn_t, _ = self.cross_attention(query=q, key=k, value=v)
            # (target_channels, batch, d_model)

            # Reshape: (batch, target_channels, d_model)
            attn_t = attn_t.permute(1, 0, 2)

            # Norm + residual
            attn_t = self.norm(attn_t + queries)

            # Project to scalar: (batch, target_channels, d_model) → (batch, target_channels, 1)
            out_t = self.output_proj(attn_t).squeeze(-1)  # (batch, target_channels)

            output_time.append(out_t)

        # Stack across time: (batch, target_channels, time)
        output = torch.stack(output_time, dim=2)

        return output


class ZeroPadChannelAdapter(nn.Module):
    """Simple zero-padding channel adapter (baseline).

    Pads input channels with zeros to reach target channel count.
    Much simpler than Perceiver but doesn't learn channel relationships.

    Args:
        target_channels: Target number of channels (e.g., 128)

    Example:
        >>> adapter = ZeroPadChannelAdapter(target_channels=128)
        >>> tuh_input = torch.randn(32, 21, 200)   # TUH: 21 channels
        >>> output = adapter(tuh_input)  # (32, 128, 200) - last 107 channels are zeros
    """

    def __init__(self, target_channels: int = 128):
        super().__init__()
        self.target_channels = target_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero-pad input to target channels.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Output tensor (batch, target_channels, time)
        """
        batch_size, in_channels, time_steps = x.shape

        if in_channels >= self.target_channels:
            # Input has enough channels, just truncate
            return x[:, :self.target_channels, :]

        # Pad with zeros: (batch, target_channels - in_channels, time)
        padding = torch.zeros(
            batch_size,
            self.target_channels - in_channels,
            time_steps,
            device=x.device,
            dtype=x.dtype
        )

        # Concatenate: (batch, in_channels + padding, time)
        return torch.cat([x, padding], dim=1)
