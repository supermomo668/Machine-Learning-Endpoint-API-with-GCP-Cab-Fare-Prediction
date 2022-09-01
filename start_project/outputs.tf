# output variables is a way to organize data to be easily queried and
# to show them back to the Terraform user.
#
# As a user of Terraform, you may be interested in a few important values,
# e.g. a load balancer IP, VPN address, etc.
#
# Outputs are a way to tell Terraform what data is important.
# This data is outputted when "terraform apply" is called,
# and can be queried using the "terraform output" command.

# Modules encapsulate their resources. A resource in one module cannot directly
# depend on resources or attributes in other modules, unless those are exported
# through outputs. These outputs can be referenced in other places in your
# configuration, for example: "${module.mysql-db.instance_address}"

output workspace_instance_guide {
  value = "Please open http://${module.workspace-vm.instance_address} in your web browser and select Machine Learning Submitter.\nNote that it may take 1-2 minutes before you can access it.\nAfter the installation finishes, SSH into the instance using:\ngcloud compute --project ${module.workspace-vm.project} ssh --zone ${module.workspace-vm.zone} clouduser@${module.workspace-vm.instance_name}\n"
}
