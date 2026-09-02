output "instance_id" {
  description = "Use this as the SSH host name in the README configuration."
  value       = aws_instance.this.id
}

output "ssh_config" {
  description = "Paste this block into ~/.ssh/config, then run ssh mlx-h100."
  value       = <<-EOT
    Host mlx-h100
      HostName ${aws_instance.this.id}
      User ubuntu
      IdentityFile ~/.ssh/mlx-h100
      IdentitiesOnly yes
      ProxyCommand sh -c "aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
  EOT
}
