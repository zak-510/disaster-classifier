from typing import Dict, Any
import logging
import os

def validate_inputs(config: Dict[str, Any], test_mode: bool = False) -> bool:
    """Validate input configuration.
    
    Args:
        config: Configuration dictionary
        test_mode: If True, skip existence checks for paths
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check required sections
        required_sections = ['data', 'models', 'logging']
        for section in required_sections:
            if section not in config:
                logging.error(f"Missing required section: {section}")
                return False
        
        # Check data section
        data_config = config['data']
        if 'input_dir' not in data_config:
            logging.error("Missing input_dir in data section")
            return False
        
        # Skip path existence checks in test mode
        if not test_mode:
            input_dir = data_config['input_dir']
            if not os.path.exists(input_dir):
                logging.error(f"Input directory does not exist: {input_dir}")
                return False
        
        # Check models section
        models_config = config['models']
        required_models = ['localization', 'damage']
        for model in required_models:
            if model not in models_config:
                logging.error(f"Missing {model} model configuration")
                return False
            
            model_config = models_config[model]
            if 'checkpoint' not in model_config:
                logging.error(f"Missing checkpoint path for {model} model")
                return False
            
            # Skip path existence checks in test mode
            if not test_mode:
                checkpoint = model_config['checkpoint']
                if not os.path.exists(checkpoint):
                    logging.error(f"Model checkpoint does not exist: {checkpoint}")
                    return False
            
            # Check damage levels for damage model
            if model == 'damage' and 'damage_levels' not in model_config:
                logging.error("Missing damage_levels in damage model configuration")
                return False
        
        # Check logging section
        logging_config = config['logging']
        if 'level' not in logging_config:
            logging.error("Missing log level in logging section")
            return False
        
        # Check inference section if present
        if 'inference' in config:
            inference_config = config['inference']
            if 'batch_size' not in inference_config:
                logging.error("Missing batch_size in inference section")
                return False
            if 'min_confidence' not in inference_config:
                logging.error("Missing min_confidence in inference section")
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating configuration: {e}")
        return False 