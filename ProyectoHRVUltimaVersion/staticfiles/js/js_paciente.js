document.addEventListener('DOMContentLoaded', function () {
    // Agrega interactividad a los botones de eliminación
    const deleteButtons = document.querySelectorAll('.btn-danger');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function (e) {
            const confirmed = confirm('¿Estás seguro de eliminar este registro?');
            if (!confirmed) {
                e.preventDefault(); // Previene la acción si el usuario no confirma
            }
        });
    });
});
