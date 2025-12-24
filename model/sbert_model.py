"""
SBERT (Sentence-BERT) Implementation in Jittor

Modular architecture inspired by sentence-transformers:
- Encoder: BertModel or other transformer models
- Pooling: mean, cls, max pooling strategies
- Head: optional projection heads (none, linear, mlp, classification)
"""

from typing import Dict, Optional, Literal
import jittor as jt
from jittor import nn

# Import encoder
try:
    from .bert_model import BertConfig, BertModel
except ImportError:
    from bert_model import BertConfig, BertModel

# Import heads
try:
    from ..heads import IdentityHead, LinearHead, MLPHead, ClassificationHead
except ImportError:
    import sys
    from pathlib import Path
    # Add parent directory to path for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from heads import IdentityHead, LinearHead, MLPHead, ClassificationHead

from jittor_utils.load_pytorch import load_pytorch


# ============================================================================
# Pooling Module
# ============================================================================

class Pooling(nn.Module):
    """
    Pooling layer to convert token embeddings to sentence embeddings.

    Supports:
    - mean: Mean pooling over all tokens (excluding padding)
    - cls: Use [CLS] token embedding
    - max: Max pooling over all tokens
    """

    def __init__(self, pooling_mode: Literal['mean', 'cls', 'max'] = 'mean'):
        super().__init__()
        self.pooling_mode = pooling_mode

        if pooling_mode not in ['mean', 'cls', 'max']:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}. Choose from 'mean', 'cls', 'max'.")

    def execute(self, token_embeddings: jt.Var, attention_mask: jt.Var) -> jt.Var:
        """
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            sentence_embeddings: [batch_size, hidden_size]
        """
        if self.pooling_mode == 'cls':
            # Use [CLS] token (first token)
            return token_embeddings[:, 0, :]

        elif self.pooling_mode == 'mean':
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            sum_embeddings = jt.sum(token_embeddings * mask, dim=1)
            sum_mask = jt.maximum(jt.sum(mask, dim=1), jt.array(1e-9))
            return sum_embeddings / sum_mask

        elif self.pooling_mode == 'max':
            # Max pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            # Set padding tokens to large negative value
            masked_embeddings = token_embeddings.clone()
            masked_embeddings = jt.where(
                mask.expand(token_embeddings.shape) > 0,
                token_embeddings,
                jt.array(-1e9)
            )
            return jt.max(masked_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")


# ============================================================================
# Main SBERT Model
# ============================================================================

class SBERTJittor(nn.Module):
    """
    Sentence-BERT implementation in Jittor.

    Modular architecture:
    - Encoder: Transformer model (BERT, RoBERTa, etc.)
    - Pooling: Convert token embeddings to sentence embeddings
    - Head: Optional projection layer

    Example:
        # Basic SBERT with mean pooling
        model = SBERTJittor('bert-base-uncased', pooling='mean')

        # With linear projection head
        model = SBERTJittor('bert-base-uncased', pooling='mean',
                           head_type='linear', output_dim=256)

        # With MLP projection head
        model = SBERTJittor('bert-base-uncased', pooling='mean',
                           head_type='mlp', output_dim=128)
    """

    def __init__(
        self,
        encoder_name: str = 'bert-base-uncased',
        pooling: Literal['mean', 'cls', 'max'] = 'mean',
        head_type: Literal['none', 'linear', 'mlp'] = 'none',
        output_dim: Optional[int] = None,
        config: Optional[BertConfig] = None,
        checkpoint_path: Optional[str] = None,
        **head_kwargs
    ):
        """
        Args:
            encoder_name: Name of the encoder model
            pooling: Pooling strategy ('mean', 'cls', 'max')
            head_type: Type of projection head ('none', 'linear', 'mlp')
            output_dim: Output dimension for projection head (required if head_type != 'none')
            config: BertConfig object (if None, will be created from encoder_name)
            checkpoint_path: Path to PyTorch checkpoint to load
            **head_kwargs: Additional arguments for projection head (e.g., hidden_dim, num_layers)
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.pooling_mode = pooling
        self.head_type = head_type

        # Build config if not provided
        if config is None:
            config = self._build_config(encoder_name)
        self.config = config

        # 1. Encoder module
        # Disable BertPooler since SBERT uses its own pooling module.
        self.encoder = BertModel(config, add_pooling_layer=False)

        # 2. Pooling module
        self.pooling = Pooling(pooling_mode=pooling)

        # 3. Projection head module
        hidden_size = config.hidden_size
        self.head = self._build_head(head_type, hidden_size, output_dim, **head_kwargs)

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        print(f"SBERTJittor initialized:")
        print(f"  Encoder: {encoder_name}")
        print(f"  Pooling: {pooling}")
        print(f"  Head: {head_type}")
        print(f"  Output dim: {self.output_dim}")

    @property
    def output_dim(self) -> int:
        """Output dimension of the sentence embeddings"""
        return self.head.output_dim

    @staticmethod
    def _build_config(encoder_name: str) -> BertConfig:
        """Build BertConfig from encoder name"""
        encoder_lower = encoder_name.lower()

        # RoBERTa models
        if 'roberta' in encoder_lower:
            if 'base' in encoder_lower:
                return BertConfig(
                    vocab_size=50265,  # RoBERTa vocab size
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=514,  # RoBERTa uses 514
                    type_vocab_size=1,  # RoBERTa doesn't use token type IDs
                    initializer_range=0.02,
                )
            elif 'large' in encoder_lower:
                return BertConfig(
                    vocab_size=50265,
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=514,
                    type_vocab_size=1,
                    initializer_range=0.02,
                )

        # BERT models
        elif 'bert' in encoder_lower:
            if 'base' in encoder_lower:
                return BertConfig(
                    vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=512,
                    type_vocab_size=2,
                    initializer_range=0.02,
                )
            elif 'large' in encoder_lower:
                return BertConfig(
                    vocab_size=30522,
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=512,
                    type_vocab_size=2,
                    initializer_range=0.02,
                )

        # Default to BERT base
        return BertConfig()

    def _build_head(
        self,
        head_type: str,
        input_dim: int,
        output_dim: Optional[int],
        **kwargs
    ) -> nn.Module:
        """Build projection head based on type"""
        if head_type == 'none':
            return IdentityHead(input_dim)

        elif head_type == 'linear':
            if output_dim is None:
                raise ValueError("output_dim is required for linear head")
            return LinearHead(input_dim, output_dim)

        elif head_type == 'mlp':
            if output_dim is None:
                raise ValueError("output_dim is required for mlp head")
            return MLPHead(input_dim, output_dim, **kwargs)

        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def encode(
        self,
        input_ids: jt.Var,
        attention_mask: jt.Var,
        token_type_ids: Optional[jt.Var] = None
    ) -> jt.Var:
        """
        Encode input to sentence embeddings.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)

        Returns:
            sentence_embeddings: [batch_size, output_dim]
        """
        # 1. Get token embeddings from encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        token_embeddings = outputs['last_hidden_state']  # [batch, seq_len, hidden]

        # 2. Pool to sentence embeddings
        sentence_embeddings = self.pooling(token_embeddings, attention_mask)

        # 3. Apply projection head
        sentence_embeddings = self.head(sentence_embeddings)

        return sentence_embeddings

    def execute(self, batch: Dict[str, jt.Var]) -> jt.Var:
        """
        Forward pass - encode a batch.

        Args:
            batch: Dictionary containing:
                - 'input_ids': [batch_size, seq_len]
                - 'attention_mask': [batch_size, seq_len]
                - 'token_type_ids': [batch_size, seq_len] (optional)

        Returns:
            sentence_embeddings: [batch_size, output_dim]
        """
        return self.encode(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None)
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load encoder weights from PyTorch checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = load_pytorch(checkpoint_path)
        self.encoder.load_state_dict(state_dict)
        print("Checkpoint loaded successfully")

    def save(self, save_path: str):
        """Save model state"""
        jt.save({
            'encoder': self.encoder.state_dict(),
            'pooling_mode': self.pooling_mode,
            'head_type': self.head_type,
            'head_state': self.head.state_dict() if self.head_type != 'none' else None,
            'config': self.config.__dict__,
        }, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path: str):
        """Load model state"""
        checkpoint = jt.load(load_path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        if checkpoint['head_state'] is not None:
            self.head.load_state_dict(checkpoint['head_state'])
        print(f"Model loaded from {load_path}")


# ============================================================================
# SBERT with Classification Head (for NLI training)
# ============================================================================

class SBERTWithClassification(nn.Module):
    """
    SBERT with classification head for NLI training.

    Uses two sentence encodings and concatenates them with element-wise difference
    for 3-way classification (entailment, neutral, contradiction).
    """

    def __init__(
        self,
        encoder_name: str = 'bert-base-uncased',
        pooling: str = 'mean',
        num_labels: int = 3,
        config: Optional[BertConfig] = None,
        checkpoint_path: Optional[str] = None
    ):
        super().__init__()

        # SBERT encoder (no projection head for NLI training)
        self.sbert = SBERTJittor(
            encoder_name=encoder_name,
            pooling=pooling,
            head_type='none',
            config=config,
            checkpoint_path=checkpoint_path
        )

        # Classification head: [u, v, |u-v|] -> logits
        self.classifier = ClassificationHead(
            hidden_size=self.sbert.output_dim,
            num_labels=num_labels
        )

        self.num_labels = num_labels

    def encode(
        self,
        input_ids: jt.Var,
        attention_mask: jt.Var,
        token_type_ids: Optional[jt.Var] = None
    ) -> jt.Var:
        """Encode sentences (delegates to SBERT)"""
        return self.sbert.encode(input_ids, attention_mask, token_type_ids)

    def execute(self, batch: Dict[str, jt.Var]) -> jt.Var:
        """
        Forward pass for NLI classification.

        Args:
            batch: Dictionary containing:
                - 'input_ids_a', 'attention_mask_a': First sentence
                - 'input_ids_b', 'attention_mask_b': Second sentence
                - 'token_type_ids_a', 'token_type_ids_b': (optional)

        Returns:
            logits: [batch_size, num_labels]
        """
        # Encode both sentences
        emb_a = self.encode(
            batch['input_ids_a'],
            batch['attention_mask_a'],
            batch.get('token_type_ids_a', None)
        )
        emb_b = self.encode(
            batch['input_ids_b'],
            batch['attention_mask_b'],
            batch.get('token_type_ids_b', None)
        )

        # Use classification head
        logits = self.classifier(emb_a, emb_b)

        return logits

    def load_checkpoint(self, checkpoint_path: str):
        """Load encoder checkpoint (delegates to SBERT)"""
        self.sbert.load_checkpoint(checkpoint_path)


# ============================================================================
# Helper function for creating models
# ============================================================================

def create_sbert_model(
    encoder_name: str = 'bert-base-uncased',
    pooling: str = 'mean',
    head_type: str = 'none',
    output_dim: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    for_training: bool = False,
    num_labels: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create SBERT models.

    Args:
        encoder_name: Encoder model name
        pooling: Pooling strategy
        head_type: Projection head type
        output_dim: Output dimension for projection
        checkpoint_path: Path to pretrained weights
        for_training: If True, return SBERTWithClassification
        num_labels: Number of labels for classification
        **kwargs: Additional arguments for head

    Returns:
        SBERT model (either SBERTJittor or SBERTWithClassification)
    """
    if for_training:
        return SBERTWithClassification(
            encoder_name=encoder_name,
            pooling=pooling,
            num_labels=num_labels,
            checkpoint_path=checkpoint_path
        )
    else:
        return SBERTJittor(
            encoder_name=encoder_name,
            pooling=pooling,
            head_type=head_type,
            output_dim=output_dim,
            checkpoint_path=checkpoint_path,
            **kwargs
        )


if __name__ == "__main__":
    # Test different configurations
    print("=" * 70)
    print("Testing SBERT Jittor Models")
    print("=" * 70)

    # 1. Basic SBERT with mean pooling (BERT)
    print("\n1. BERT-base SBERT (mean pooling, no head)")
    model1 = SBERTJittor('bert-base-uncased', pooling='mean', head_type='none')
    print(f"Output dim: {model1.output_dim}")

    # 2. RoBERTa SBERT
    print("\n2. RoBERTa-base SBERT (mean pooling, no head)")
    model2 = SBERTJittor('roberta-base', pooling='mean', head_type='none')
    print(f"Output dim: {model2.output_dim}")
    print(f"Vocab size: {model2.config.vocab_size}")
    print(f"Max position embeddings: {model2.config.max_position_embeddings}")

    # 3. SBERT with linear projection
    print("\n3. SBERT with linear projection head")
    model3 = SBERTJittor('bert-base-uncased', pooling='mean', head_type='linear', output_dim=256)
    print(f"Output dim: {model3.output_dim}")

    # 4. SBERT with MLP projection
    print("\n4. SBERT with MLP projection head")
    model4 = SBERTJittor('bert-base-uncased', pooling='mean', head_type='mlp', output_dim=128, num_layers=2)
    print(f"Output dim: {model4.output_dim}")

    # 5. SBERT for training (with classification head)
    print("\n5. SBERT with classification head (for NLI training)")
    model5 = SBERTWithClassification('bert-base-uncased', pooling='mean', num_labels=3)
    print(f"Encoder output dim: {model5.sbert.output_dim}")
    print(f"Num labels: {model5.num_labels}")

    # 6. RoBERTa for training
    print("\n6. RoBERTa with classification head (for NLI training)")
    model6 = SBERTWithClassification('roberta-base', pooling='mean', num_labels=3)
    print(f"Encoder output dim: {model6.sbert.output_dim}")
    print(f"Num labels: {model6.num_labels}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
