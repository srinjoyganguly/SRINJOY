function restartHandler(){
    Jupyter.notebook.restart_kernel();
}; 
jQuery("#restart").click(restartHandler);