terraform {
  required_version = ">= 1.9.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

data "aws_ssm_parameter" "gpu_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}

resource "aws_vpc" "this" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "mlx-h100-vpc"
  }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "mlx-h100-igw"
  }
}

resource "aws_subnet" "this" {
  vpc_id                  = aws_vpc.this.id
  availability_zone       = "ap-northeast-1a"
  cidr_block              = "10.0.0.0/24"
  map_public_ip_on_launch = true

  tags = {
    Name = "mlx-h100-subnet"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = {
    Name = "mlx-h100-public"
  }
}

resource "aws_route_table_association" "this" {
  subnet_id      = aws_subnet.this.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "instance" {
  name        = "mlx-h100-instance"
  description = "No inbound access; outbound only for SSM and package installation"
  vpc_id      = aws_vpc.this.id

  egress {
    description = "HTTPS and package access"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlx-h100-instance"
  }
}

resource "aws_iam_role" "instance" {
  name = "mlx-h100-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "mlx-h100-instance"
  role = aws_iam_role.instance.name
}

resource "aws_key_pair" "owner" {
  key_name   = "mlx-h100-owner"
  public_key = file("${path.module}/mlx-h100.pub")
}

resource "aws_instance" "this" {
  ami                         = data.aws_ssm_parameter.gpu_ami.value
  instance_type               = "g6e.4xlarge"
  subnet_id                   = aws_subnet.this.id
  vpc_security_group_ids      = [aws_security_group.instance.id]
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  key_name                    = aws_key_pair.owner.key_name
  associate_public_ip_address = true

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {})

  tags = {
    Name = "mlx-h100"
  }
}
