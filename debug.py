import warp as wp
@wp.kernel
def wp_preprocess(
    x: int,
    ss: int
):
    # Your simplified kernel code here
    pass

num_points = 10
# Then when launching:
wp.launch(
    kernel=wp_preprocess,
    dim=3,
    inputs=[0, 2,]
)