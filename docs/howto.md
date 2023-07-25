# Developer

Developers can use this page to update docs!

## How To Add Custom Pages to RTD pages

1. Create a `.md` new file in the `docs` folder

2. Add your markdown text that will be static on that page

3. Open the `mkdocs.yml` file and add the `.md` file that you created in the following section, so that you can control where it goes in the table of contents.

```
nav:
    - Home: 'index.md'
    - API Reference: 'api.md'
    - Developer: 'howto.md'
```

For more information and styling of pages, you can find more information [here.](https://mkdocs.readthedocs.io/en/0.14.0/user-guide/writing-your-docs/#configure-pages-and-navigation)

## How to add modules to the API

1. Open the `api.md` file in the `docs` folder

2. Place which module you want to add to the file in the following format using 
`::: moduleFolder.moduleName`
```
# API Reference

::: riid

::: riid.data

::: riid.gadras

::: riid.models

::: riid.anomaly

::: riid.visualize

```
### Autogenerate

You can autogenerate the api modules from folders that you pick. Currently it is only generated from the folders, `riid`, `tests`, `examples`. In the `docs` folder you can run the following command and it will generate the `api.md` file for you based on the modules in the preivously stated folders.

```
python .\generate_api_md.py
```


## API reference configurations

There are multiple styles we can use for our api. You can find more about those configurations [here.](https://mkdocstrings.github.io/python/reference/mkdocstrings_handlers/python/handler/)

```
default_config: dict = {
    "docstring_style": "google",
    "docstring_options": {},
    "show_symbol_type_heading": False,
    "show_symbol_type_toc": False,
    "show_root_heading": False,
    "show_root_toc_entry": True,
    "show_root_full_path": True,
    "show_root_members_full_path": False,
    "show_object_full_path": False,
    "show_category_heading": False,
    "show_if_no_docstring": False,
    "show_signature": True,
    "show_signature_annotations": False,
    "signature_crossrefs": False,
    "separate_signature": False,
    "line_length": 60,
    "merge_init_into_class": False,
    "show_docstring_attributes": True,
    "show_docstring_description": True,
    "show_docstring_examples": True,
    "show_docstring_other_parameters": True,
    "show_docstring_parameters": True,
    "show_docstring_raises": True,
    "show_docstring_receives": True,
    "show_docstring_returns": True,
    "show_docstring_warns": True,
    "show_docstring_yields": True,
    "show_source": True,
    "show_bases": True,
    "show_submodules": False,
    "group_by_category": True,
    "heading_level": 2,
    "members_order": rendering.Order.alphabetical.value,
    "docstring_section_style": "table",
    "members": None,
    "inherited_members": False,
    "filters": ["!^_[^_]"],
    "annotations_path": "brief",
    "preload_modules": None,
    "load_external_modules": False,
    "allow_inspection": True,
}
```