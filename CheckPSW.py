import torch
import videoseal

def inspect_module_parameters(target_module, module_name="PixelSeal"):
    print("=" * 70)
    print(f"ANALISI DETTAGLIATA PARAMETRI PER: {module_name.upper()}")
    print("=" * 70)
    
    total_params_count = 0
    trainable_params_count = 0
    frozen_params_count = 0
    
    # Dictionary to keep track of statistics for first-level submodules
    submodule_stats = {}

    for param_name, param_tensor in target_module.named_parameters():
        num_elements = param_tensor.numel()
        total_params_count += num_elements
        is_trainable = param_tensor.requires_grad
        
        if is_trainable:
            trainable_params_count += num_elements
        else:
            frozen_params_count += num_elements

        # Extract the top-level submodule name for grouped reporting
        name_parts = param_name.split('.')
        top_submodule = name_parts[0] if len(name_parts) > 0 else "root"
        
        if top_submodule not in submodule_stats:
            submodule_stats[top_submodule] = {
                "total": 0, 
                "trainable": 0, 
                "frozen": 0
            }
        
        submodule_stats[top_submodule]["total"] += num_elements
        if is_trainable:
            submodule_stats[top_submodule]["trainable"] += num_elements
        else:
            submodule_stats[top_submodule]["frozen"] += num_elements

        # Optional: uncomment the line below if you want to print every single layer tensor
        # print(f"Layer: {param_name:<65} | Elements: {num_elements:<10} | Trainable: {is_trainable}")

    # Print summary table for submodules
    print("\nRIEPILOGO STRUTTURALE PER SOTTO-MODULI:")
    print("-" * 70)
    for sub_name, info in submodule_stats.items():
        print(f" Sotto-modulo: '{sub_name}'")
        print(f"   -> Parametri Totali:    {info['total']:,}")
        print(f"   -> Parametri Trainable: {info['trainable']:,}")
        print(f"   -> Parametri Frozen:    {info['frozen']:,}")
    print("-" * 70)

    # Print global module statistics
    print(f"\nSTATISTICHE GLOBALI ({module_name}):")
    print(f"  Parametri Totali (Total):          {total_params_count:,}")
    print(f"  Parametri Addestrabili (Trainable): {trainable_params_count:,}")
    print(f"  Parametri Congelati (Frozen):       {frozen_params_count:,}")
    
    if total_params_count > 0:
        trainable_percentage = (trainable_params_count / total_params_count) * 100
        print(f"  Percentuale Addestrabile:           {trainable_percentage:.2f}%")
    print("=" * 70 + "\n")


def main():
    # Set the execution device
    execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caricamento PixelSeal sul dispositivo: {execution_device}...\n")
    
    # Load the official pre-trained PixelSeal model from Meta/VideoSeal
    pixelseal_model = videoseal.load("pixelseal").to(execution_device)
    
    # Step 1: Inspect the Watermarker / Embedder submodule if present
    if hasattr(pixelseal_model, 'embedder'):
        inspect_module_parameters(
            target_module=pixelseal_model.embedder, 
            module_name="PixelSeal - Embedder (Watermarker)"
        )
        
    # Step 2: Inspect the Detector submodule if present
    if hasattr(pixelseal_model, 'detector'):
        inspect_module_parameters(
            target_module=pixelseal_model.detector, 
            module_name="PixelSeal - Detector"
        )
        
    # Step 3: Inspect the full unified model architecture
    inspect_module_parameters(
        target_module=pixelseal_model, 
        module_name="PixelSeal - Modello Unificato Completo"
    )

if __name__ == "__main__":
    main()