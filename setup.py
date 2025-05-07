# mi_paquete_icc/setup.py

from setuptools import setup, find_packages

setup(
    name='intrinsic_compositing_clean',  # Nombre que se usará para pip install (e.g., pip install intrinsic-compositing-utilities)
                                            # Puede ser diferente al nombre del paquete importable.
                                            # Usa guiones bajos para nombres de paquete importables, guiones para nombres de distribución.
                                            # Si quieres que el nombre de PyPI sea igual al de importación, usa 'intrinsic_compositing_clean'
    version='0.1.0',                        # Versión de tu paquete (sigue SemVer: MAJOR.MINOR.PATCH)

    author='',                     # Tu nombre o el de tu organización
    author_email='tu.email@example.com',    # Tu email de contacto

    description='Una biblioteca de utilidades para composición intrínseca.', # Breve descripción

    #long_description=read_readme(),         # Descripción larga, usualmente desde un archivo README.md
    #long_description_content_type='text/markdown', # Si tu README es Markdown

    #url='https://github.com/tu_usuario/tu_repositorio', # URL del proyecto (opcional, pero bueno para GitHub)

    # Aquí es donde le dices a setuptools dónde encontrar tus paquetes.
    # find_packages() buscará automáticamente todos los paquetes (directorios con __init__.py)
    # en el directorio actual (donde está setup.py).
    # Encontrará 'intrinsic_compositing_clean' y su subpaquete 'lib'.
    packages=find_packages(where="."),      # Busca paquetes en el directorio actual ('.')
                                            # Esto es correcto para tu estructura.
                                            # Si tuvieras una carpeta 'src/' y tus paquetes dentro de ella,
                                            # usarías: package_dir={'': 'src'}, packages=find_packages(where='src')

    # Clasificadores para ayudar a la gente a encontrar tu paquete en PyPI (opcional)
    classifiers=[
        'Development Status :: 3 - Alpha', # O Beta, Production/Stable
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', # ¡Elige una licencia!
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Graphics', # Ejemplo de tópico relevante
    ],

    install_requires=[
        'torch',
        'torchvision',
        'scikit-image',
        'matplotlib',
        'kornia',
        'timm',
        'scipy',
        'opencv-python==4.11.0.86',
        'imageio'
    ],

    python_requires='>=3.7', # Versión mínima de Python requerida
)
