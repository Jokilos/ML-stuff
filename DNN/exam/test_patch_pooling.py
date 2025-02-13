
import pytest
import torch


class PatchPooling(torch.nn.Module):
    def forward(self, batch: torch.Tensor, patch_lengths: torch.Tensor) -> torch.Tensor:
        B, S, D = batch.shape
        B_1, P = patch_lengths.shape

        assert B == B_1

        ### Your code goes here ###

        # make from - to matrices
        batch_to_od = torch.cumsum(patch_lengths, dim = -1)
        batch_from_od = torch.hstack([torch.zeros(B, 1), batch_to_od]) [:, :P]

        tsize = (P, *batch.shape)
        thresh = torch.arange(S).reshape(-1, 1).tile((P, B, 1, D))

        # expand batch and match to creted mask
        mask = thresh < batch_to_od[:, :, None, None]
        mask &= thresh >= batch_from_od[:, :, None, None]

        # sum vectors picked from every sequence
        patch_pool = torch.repeat_interleave(batch, P, dim = 0).reshape(tsize)
        patch_pool[~mask] = 0
        patch_pool = torch.sum(patch_pool, dim = 2)

        # find how many vectors are contributing
        divisor = torch.sum(mask, dim = 2)

        # make sure we don't divide by 0
        patch_pool[divisor == 0] = -1
        divisor[divisor == 0] = 1

        patch_pool /= divisor
        
        return patch_pool
         
        ###########################

                


class TestPatchPooling:
    @pytest.mark.parametrize(
        "batch,patch_lengths,expected_output",
        [
            (
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                        ],
                        [
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [6.0, 6.0, 6.0, 6.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
                torch.tensor([[3, 1, 2], [4, 1, 0], [1, 0, 0]]),
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                        ],
                        [
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [6.0, 6.0, 6.0, 6.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
            )
        ],
    )
    def test_forward(
        self,
        batch: torch.Tensor,
        patch_lengths: torch.Tensor,
        expected_output: torch.Tensor,
    ) -> None:
        # given
        patch_pooling = PatchPooling()

        # when
        output = patch_pooling(batch=batch, patch_lengths=patch_lengths)

        # then
        assert torch.all(torch.isclose(output, expected_output))
