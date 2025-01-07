import os
import subprocess

def transfer_folders_to_remote(local_root_dir, crsid, machine):
    """
    Transfer train, validation, and test folders to remote scratch directory.
    """
    sets = ['train', 'validation', 'test']
    host = f"{machine}.cl.cam.ac.uk"
    scratch_dir = f"/local/scratch/{crsid}"
    
    # SSH options for Cambridge authentication
    ssh_options = "-o GSSAPIAuthentication=yes -o GSSAPIDelegateCredentials=yes -o PubkeyAuthentication=no"
    
    # Create directories in scratch space
    mkdir_cmd = f'ssh {ssh_options} {crsid}@{host} "mkdir -p {scratch_dir}/images {scratch_dir}/labels"'
    print(f"Creating remote directories in {scratch_dir}...")
    subprocess.run(mkdir_cmd, shell=True)
    
    for set_type in sets:
        img_dir = os.path.join(local_root_dir, 'images', set_type).replace('\\', '/')
        label_dir = os.path.join(local_root_dir, 'labels', set_type).replace('\\', '/')
        
        print(f"\nTransferring {set_type} set...")
        
        # Transfer with explicit authentication options
        if os.path.exists(img_dir):
            cmd = f'scp -r -o GSSAPIAuthentication=yes -o GSSAPIDelegateCredentials=yes -o PubkeyAuthentication=no "{img_dir}" {crsid}@{host}:{scratch_dir}/images/'
            print(f"Executing: {cmd}")
            subprocess.run(cmd, shell=True)
        else:
            print(f"Warning: Image directory not found: {img_dir}")
            
        if os.path.exists(label_dir):
            cmd = f'scp -r -o GSSAPIAuthentication=yes -o GSSAPIDelegateCredentials=yes -o PubkeyAuthentication=no "{label_dir}" {crsid}@{host}:{scratch_dir}/labels/'
            print(f"Executing: {cmd}")
            subprocess.run(cmd, shell=True)
        else:
            print(f"Warning: Label directory not found: {label_dir}")

def check_remote_space(crsid, machine):
    """
    Check available space in remote scratch directory.
    """
    host = f"{machine}.cl.cam.ac.uk"
    scratch_dir = f"/local/scratch/{crsid}"
    
    ssh_options = "-o GSSAPIAuthentication=yes -o GSSAPIDelegateCredentials=yes -o PubkeyAuthentication=no"
    cmd = f'ssh {ssh_options} {crsid}@{host} "df -h {scratch_dir}"'
    print("\nChecking remote storage space...")
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    # Configuration
    local_root = "M:/Unused/TauCellDL"
    crsid = "pz286"
    machine = "kiiara"
    
    print("Data Transfer to Kiiara Scratch Space")
    print("=====================================")
    print(f"Local data directory: {local_root}")
    print(f"Remote user: {crsid}")
    print(f"Remote machine: {machine}")
    print("\nBefore running this script:")
    print("1. Make sure you've run: kinit CRSID@DC.CL.CAM.AC.UK")
    print("2. Check tickets with: klist")
    
    response = input("Have you run kinit and confirmed tickets? (y/n): ")
    if response.lower() != 'y':
        print("Please run kinit first.")
        exit()
    
    # Check remote space before transfer
    check_remote_space(crsid, machine)
    
    # Confirm before proceeding
    response = input("\nProceed with transfer? (y/n): ")
    if response.lower() != 'y':
        print("Transfer cancelled.")
        exit()
    
    # Perform transfer
    transfer_folders_to_remote(local_root, crsid, machine)
    
    # Check space after transfer
    print("\nTransfer complete. Checking final storage status:")
    check_remote_space(crsid, machine)