# One L40S development instance

This creates one `g6e.4xlarge`: one NVIDIA L40S with 48 GiB VRAM. It has no inbound security-group rules. SSH uses an AWS Systems Manager (SSM) tunnel, so the instance's public IP is used only for its outbound access to SSM, Ubuntu packages, and NVIDIA's CUDA repository.

The root volume is the only EBS volume. It is configured to be deleted with the instance; there is no separate data volume.

The AMI is AWS's current Ubuntu 24.04 Deep Learning Base GPU AMI with the NVIDIA driver. At boot, user data installs Clang 18, CMake, Git, BLAS/LAPACK, CUDA 12.9, and cuDNN 9. MLX C++ can then be built with `-DMLX_BUILD_CUDA=ON`.

## Prerequisites

- Terraform, AWS CLI v2, and the AWS Session Manager plugin on this Mac.
- AWS credentials allowed to create the resources in this directory and allowed to start SSM sessions.
- The AWS account must have quota and capacity for `g6e.4xlarge` in Tokyo. This project uses `ap-northeast-1a`.

Create a local SSH key, then copy only its public half next to the Terraform files. Do not put the private key in this directory:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/mlx-h100 -C "mlx-h100"
cp ~/.ssh/mlx-h100.pub ./mlx-h100.pub
```

The Terraform is fully hard-coded for this Mac: Tokyo (`ap-northeast-1`), `ap-northeast-1a`, resource name `mlx-h100`, and `terraform/mlx-h100.pub`. Then create the instance:

```sh
terraform init
terraform apply
```

## SSH through SSM

After `apply`, copy the `ssh_config` output, or paste this into `~/.ssh/config` after replacing `INSTANCE_ID` with the `instance_id` output:

```sshconfig
Host mlx-h100
  HostName INSTANCE_ID
  User ubuntu
  IdentityFile ~/.ssh/mlx-h100
  IdentitiesOnly yes
  ProxyCommand sh -c "aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
```

Connect once SSM reports the instance as managed:

```sh
ssh mlx-h100
```

On the instance, confirm the GPU and build MLX:

```sh
nvidia-smi
git clone https://github.com/ml-explore/mlx.git
cmake -S mlx -B mlx/build -DMLX_BUILD_CUDA=ON -DMLX_BUILD_METAL=OFF
cmake --build mlx/build -j"$(nproc)"
```

Destroy the instance when finished to stop GPU charges:

```sh
terraform destroy
```
