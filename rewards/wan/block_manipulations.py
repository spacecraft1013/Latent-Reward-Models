# block_manipulation.py
import torch
import torch.nn as nn
import copy
from typing import List, Optional, Union, Literal
from dataclasses import dataclass
import logging
from wan.modules.model import WanLayerNorm, WanRMSNorm


@dataclass
class BlockInsertion:
    """Defines how to insert a new block"""
    position: int  # Insert after this position (-1 for beginning)
    source_blocks: Union[int, List[int]]  # Block index(es) to copy from
    init_strategy: Literal['copy', 'average', 'zero', 'random'] = 'copy'


def apply_block_manipulations(model, blocks_to_skip: List[int], blocks_to_insert: List[BlockInsertion]):
    """
    Apply block skipping and insertion after model is loaded.

    Args:
        model: The WanModel instance
        blocks_to_skip: List of block indices to remove
        blocks_to_insert: List of BlockInsertion specifications

    Returns:
        Modified model with blocks skipped/inserted
    """
    # Get the original blocks
    original_blocks = list(model.blocks)
    num_original = len(original_blocks)

    logging.info(f"Original model has {num_original} blocks")

    # Step 1: Create mapping from original to new indices (accounting for skips)
    orig_to_new = {}
    new_blocks = []
    new_idx = 0

    for orig_idx in range(num_original):
        if orig_idx not in blocks_to_skip:
            orig_to_new[orig_idx] = new_idx
            new_blocks.append(original_blocks[orig_idx])
            new_idx += 1

    logging.info(f"After skipping {len(blocks_to_skip)} blocks, have {len(new_blocks)} blocks")

    # Step 2: Handle insertions
    if blocks_to_insert:
        # Sort insertions by position
        insertions = sorted(blocks_to_insert, key=lambda x: x.position)

        final_blocks = []
        ptr = 0

        for insertion in insertions:
            # Adjust insertion position based on skipped blocks
            adjusted_position = insertion.position
            if adjusted_position >= 0:
                # Count how many blocks before this position were skipped
                skipped_before = sum(1 for skip_idx in blocks_to_skip if skip_idx <= insertion.position)
                adjusted_position = insertion.position - skipped_before

            # Add blocks up to insertion point
            while ptr <= adjusted_position and ptr < len(new_blocks):
                final_blocks.append(new_blocks[ptr])
                ptr += 1

            # Create new block based on insertion spec
            new_block = create_inserted_block(original_blocks, model, insertion, orig_to_new)
            final_blocks.append(new_block)

            logging.info(f"Inserted block at position {len(final_blocks)-1} from source {insertion.source_blocks}")

        # Add remaining blocks
        while ptr < len(new_blocks):
            final_blocks.append(new_blocks[ptr])
            ptr += 1

        new_blocks = final_blocks

    # Step 3: Replace the model's blocks
    model.blocks = nn.ModuleList(new_blocks)

    # Update model metadata
    model.num_layers = len(model.blocks)
    # Create new block mapping for debugging
    model.manipulated_block_mapping = {}
    for i, block in enumerate(model.blocks):
        # Try to find which original block this came from
        for orig_idx, orig_block in enumerate(original_blocks):
            if block is orig_block:
                model.manipulated_block_mapping[i] = orig_idx
                break
        else:
            model.manipulated_block_mapping[i] = f"inserted at {i}"

    logging.info(f"Final model has {len(model.blocks)} blocks")
    logging.info(f"Block mapping: {model.manipulated_block_mapping}")

    return model

def cleanup_skipped_blocks(skipped_blocks):
    """Properly cleanup skipped blocks to free memory."""
    for block in skipped_blocks:
        # Move to CPU first to free GPU memory
        block.cpu()

        # Delete all parameters
        for param in block.parameters():
            del param

        # Clear any buffers
        for buffer in block.buffers():
            del buffer

    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()


## Block Manipulations with Norm
class BridgeAffine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))
    def forward(self, x, *args, **kwargs):
        return x * self.scale + self.bias

class BridgeLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps, elementwise_affine=elementwise_affine)
    def forward(self, x, *args, **kwargs):
        return self.norm(x)

class BridgeRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, with_bias=True):
        super().__init__()
        self.norm = WanRMSNorm(dim, eps)
        self.bias = nn.Parameter(torch.zeros(dim)) if with_bias else None
    def forward(self, x, *args, **kwargs):
        y = self.norm(x)
        return y + self.bias if self.bias is not None else y


def apply_block_manipulations_norm(model, blocks_to_skip, blocks_to_insert,
                              bridge_strategy: str = "none",
                              bridge_eps: float = 1e-5,
                              bridge_ln_affine: bool = True):
    # Get the original blocks
    original_blocks = list(model.blocks)
    num_original = len(original_blocks)

    logging.info(f"Original model has {num_original} blocks")

    # Step 1: Create mapping from original to new indices (accounting for skips)
    orig_to_new = {}
    new_blocks = []
    new_idx = 0

    for orig_idx in range(num_original):
        if orig_idx not in blocks_to_skip:
            orig_to_new[orig_idx] = new_idx
            new_blocks.append(original_blocks[orig_idx])
            new_idx += 1

    logging.info(f"After skipping {len(blocks_to_skip)} blocks, have {len(new_blocks)} blocks")

    # Step 3: Replace the model's blocks
    model.blocks = nn.ModuleList(new_blocks)

    # Update model metadata
    model.num_layers = len(model.blocks)
    # Create new block mapping for debugging
    model.manipulated_block_mapping = {}
    for i, block in enumerate(model.blocks):
        # Try to find which original block this came from
        for orig_idx, orig_block in enumerate(original_blocks):
            if block is orig_block:
                model.manipulated_block_mapping[i] = orig_idx
                break
        else:
            model.manipulated_block_mapping[i] = f"inserted at {i}"
    logging.info("Setting up bridge norms")

    ## Setup Bridges
    bridge_positions = []
    if bridge_strategy != "none":
        final_blocks = nn.ModuleList()
        for i, blk in enumerate(new_blocks):
            if i>0:
                prev_og = model.manipulated_block_mapping[i-1]
                curr_og = model.manipulated_block_mapping[i]
                print(f"Checking bridge between {prev_og} and {curr_og}")
                if isinstance(prev_og, int) and isinstance(curr_og, int) and (curr_og - prev_og > 1):
                    if bridge_strategy == "affine":
                        bridge = BridgeAffine(blk.dim)
                    elif bridge_strategy == "layernorm":
                        bridge = BridgeLayerNorm(blk.dim, eps=bridge_eps, elementwise_affine=bridge_ln_affine)
                    elif bridge_strategy == "rmsnorm":
                        bridge = BridgeRMSNorm(blk.dim, eps=bridge_eps, with_bias=True)
                    else:
                        raise ValueError(f"Unsupported bridge {bridge_strategy}")
                    logging.info(f"Added bridge ({bridge_strategy}) norm between blocks {prev_og} and {curr_og}")
                    final_blocks.append(bridge)
                    bridge_positions.append((len(final_blocks)-1, prev_og, curr_og))
            final_blocks.append(blk)
        model.blocks = final_blocks
    else:
        model.blocks = nn.ModuleList(new_blocks)

    model.bridge_meta = {"positions": bridge_positions, "strategy": bridge_strategy}
    model.num_layers = len(model.blocks)

    logging.info(f"Final model has {len(model.blocks)} blocks")
    logging.info(f"Block mapping: {model.manipulated_block_mapping}")

    return model


def create_inserted_block(original_blocks, model, insertion: BlockInsertion, orig_to_new):
    """Create a new block based on insertion specification."""
    # Get reference blocks
    source_indices = insertion.source_blocks if isinstance(insertion.source_blocks, list) else [insertion.source_blocks]
    source_blocks = [original_blocks[idx] for idx in source_indices]

    # Create new block with same configuration
    first_block = source_blocks[0]
    new_block = type(first_block)(
        first_block.cross_attn_type,
        first_block.dim,
        first_block.ffn_dim,
        first_block.num_heads,
        first_block.window_size,
        first_block.qk_norm,
        first_block.cross_attn_norm,
        first_block.eps
    )

    # Initialize weights based on strategy
    if insertion.init_strategy == 'copy':
        new_block.load_state_dict(source_blocks[0].state_dict())

    elif insertion.init_strategy == 'average':
        # Average weights from all source blocks
        avg_state = {}
        state_dicts = [block.state_dict() for block in source_blocks]

        for key in state_dicts[0]:
            tensors = [sd[key] for sd in state_dicts]
            avg_state[key] = torch.stack(tensors).mean(dim=0)

        new_block.load_state_dict(avg_state)

    elif insertion.init_strategy == 'zero':
        # Zero initialize
        for param in new_block.parameters():
            param.data.zero_()

    elif insertion.init_strategy == 'random':
        # Keep random initialization
        pass

    else:
        raise ValueError(f"Unknown init strategy: {insertion.init_strategy}")

    return new_block

def print_model_structure(model, blocks_to_skip, blocks_to_insert, original_num_blocks=30):
    """
    Generate a concise string representation of model structure after manipulation.

    Returns:
        str: Model structure visualization
    """
    lines = []
    lines.append("=== Model Block Structure ===")
    lines.append(f"Original blocks: {original_num_blocks}")
    lines.append(f"Blocks to skip: {sorted(blocks_to_skip)}")
    lines.append(f"Insertions: {[(i.position, i.source_blocks, i.init_strategy) for i in blocks_to_insert]}")
    lines.append(f"Final block count: {len(model.blocks)}")
    lines.append("\nBlock Mapping:")
    lines.append("Final Idx -> Original Idx/Source")
    lines.append("-" * 35)

    # Create a detailed mapping
    if hasattr(model, 'manipulated_block_mapping'):
        for final_idx in range(len(model.blocks)):
            source = model.manipulated_block_mapping.get(final_idx, 'unknown')
            lines.append(f"Block {final_idx:3d} <- {source}")
    else:
        # Try to infer the mapping
        lines.append("(Block mapping not available)")

    lines.append("\nStructure Visualization:")
    lines.append("-" * 50)

    # Visual representation
    visual = []
    for i in range(original_num_blocks):
        if i in blocks_to_skip:
            visual.append(f"[{i:2d}:SKIP]")
        else:
            visual.append(f"[{i:2d}:KEEP]")

    lines.append("Original: " + " ".join(visual[:16]))
    lines.append("          " + " ".join(visual[16:]))

    # Show final structure with insertions marked
    lines.append("\nFinal Structure:")
    final_visual = []
    for i in range(len(model.blocks)):
        if hasattr(model, 'manipulated_block_mapping'):
            source = model.manipulated_block_mapping.get(i, 'unknown')
            if isinstance(source, int):
                final_visual.append(f"[{i:2d}<-{source:2d}]")
            else:
                final_visual.append(f"[{i:2d}:NEW]")
        else:
            final_visual.append(f"[{i:2d}]")

    # Print in rows of 10 for readability
    for i in range(0, len(final_visual), 10):
        lines.append("         " + " ".join(final_visual[i:i+10]))

    lines.append("=== End Model Structure ===\n")

    return "\n".join(lines)

def test_block_manipulations():
    """Unit tests for block manipulation functions."""
    import unittest

    class TestBlockManipulation(unittest.TestCase):

        def setUp(self):
            """Create a mock model for testing."""
            # Create a simple mock block class
            class MockBlock(nn.Module):
                def __init__(self, block_id):
                    super().__init__()
                    self.block_id = block_id
                    self.weight = nn.Parameter(torch.randn(10, 10))
                    self.bias = nn.Parameter(torch.randn(10))
                    # Mock attributes
                    self.cross_attn_type = 't2v_cross_attn'
                    self.dim = 10
                    self.ffn_dim = 40
                    self.num_heads = 2
                    self.window_size = (-1, -1)
                    self.qk_norm = True
                    self.cross_attn_norm = True
                    self.eps = 1e-6

            # Create mock model
            class MockModel(nn.Module):
                def __init__(self, num_blocks=30):
                    super().__init__()
                    self.blocks = nn.ModuleList([MockBlock(i) for i in range(num_blocks)])
                    self.num_layers = num_blocks

            self.MockBlock = MockBlock
            self.MockModel = MockModel

        def test_skip_blocks(self):
            """Test that blocks are correctly skipped."""
            model = self.MockModel(10)
            original_blocks = list(model.blocks)

            blocks_to_skip = [2, 5, 7]
            model = apply_block_manipulations(model, blocks_to_skip, [])

            # Check correct number of blocks
            self.assertEqual(len(model.blocks), 7)

            # Check that remaining blocks are correct
            remaining_ids = [b.block_id for b in model.blocks]
            expected_ids = [0, 1, 3, 4, 6, 8, 9]
            self.assertEqual(remaining_ids, expected_ids)

        def test_copy_insertion(self):
            """Test block insertion with copy strategy."""
            model = self.MockModel(5)
            original_weight = model.blocks[2].weight.data.clone()
            original_bias = model.blocks[2].bias.data.clone()

            insertion = BlockInsertion(position=3, source_blocks=2, init_strategy='copy')
            model = apply_block_manipulations(model, [], [insertion])

            # Check correct number of blocks
            self.assertEqual(len(model.blocks), 6)

            # Check that inserted block has same weights as source
            inserted_weight = model.blocks[4].weight.data
            inserted_bias = model.blocks[4].bias.data

            self.assertTrue(torch.allclose(inserted_weight, original_weight))
            self.assertTrue(torch.allclose(inserted_bias, original_bias))

        def test_average_insertion(self):
            """Test block insertion with average strategy."""
            model = self.MockModel(5)
            weight1 = model.blocks[1].weight.data.clone()
            weight2 = model.blocks[2].weight.data.clone()
            expected_avg = (weight1 + weight2) / 2

            insertion = BlockInsertion(position=2, source_blocks=[1, 2], init_strategy='average')
            model = apply_block_manipulations(model, [], [insertion])

            # Check that inserted block has averaged weights
            inserted_weight = model.blocks[3].weight.data
            self.assertTrue(torch.allclose(inserted_weight, expected_avg, atol=1e-6))

        def test_zero_insertion(self):
            """Test block insertion with zero strategy."""
            model = self.MockModel(3)

            insertion = BlockInsertion(position=1, source_blocks=0, init_strategy='zero')
            model = apply_block_manipulations(model, [], [insertion])

            # Check that inserted block has zero weights
            inserted_weight = model.blocks[2].weight.data
            inserted_bias = model.blocks[2].bias.data

            self.assertTrue(torch.allclose(inserted_weight, torch.zeros_like(inserted_weight)))
            self.assertTrue(torch.allclose(inserted_bias, torch.zeros_like(inserted_bias)))

        def test_edge_cases(self):
            """Test edge cases for block manipulation."""
            # Test skipping all blocks
            model = self.MockModel(5)
            with self.assertRaises(ValueError):
                apply_block_manipulations(model, list(range(5)), [])

            # Test invalid block indices
            model = self.MockModel(5)
            with self.assertRaises(IndexError):
                apply_block_manipulations(model, [10], [])

            # Test insertion at beginning
            model = self.MockModel(3)
            insertion = BlockInsertion(position=-1, source_blocks=0, init_strategy='copy')
            model = apply_block_manipulations(model, [], [insertion])
            self.assertEqual(len(model.blocks), 4)

            # Verify first block is the inserted one
            self.assertTrue(torch.allclose(
                model.blocks[0].weight.data,
                model.blocks[1].weight.data  # Original block 0 is now at position 1
            ))
            """Test combined skipping and insertion."""
            model = self.MockModel(10)

            blocks_to_skip = [2, 5]
            insertions = [
                BlockInsertion(position=3, source_blocks=1, init_strategy='copy'),
                BlockInsertion(position=7, source_blocks=[6, 8], init_strategy='average')
            ]

            # Store original weights for comparison
            weight1 = model.blocks[1].weight.data.clone()
            weight6 = model.blocks[6].weight.data.clone()
            weight8 = model.blocks[8].weight.data.clone()

            model = apply_block_manipulations(model, blocks_to_skip, insertions)

            # Check correct number of blocks (10 - 2 skipped + 2 inserted = 10)
            self.assertEqual(len(model.blocks), 10)

            # Verify the structure
            block_ids = []
            for i, block in enumerate(model.blocks):
                if hasattr(block, 'block_id'):
                    block_ids.append(block.block_id)
                else:
                    block_ids.append('NEW')

            # Expected: [0, 1, 'NEW'(copy of 1), 3, 4, 6, 7, 'NEW'(avg of 6,8), 8, 9]
            self.assertEqual(block_ids[0], 0)
            self.assertEqual(block_ids[1], 1)
            self.assertEqual(block_ids[2], 'NEW')  # Inserted copy of 1
            self.assertEqual(block_ids[3], 3)
            self.assertEqual(block_ids[4], 4)
            self.assertEqual(block_ids[5], 6)
            self.assertEqual(block_ids[6], 7)
            self.assertEqual(block_ids[7], 'NEW')  # Inserted average
            self.assertEqual(block_ids[8], 8)
            self.assertEqual(block_ids[9], 9)

            # Verify weights
            self.assertTrue(torch.allclose(model.blocks[2].weight.data, weight1))
            expected_avg = (weight6 + weight8) / 2
            self.assertTrue(torch.allclose(model.blocks[7].weight.data, expected_avg, atol=1e-6))

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBlockManipulation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == "__main__":
    # Run unit tests when module is executed directly
    print("Running block manipulation unit tests...")
    success = test_block_manipulations()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)