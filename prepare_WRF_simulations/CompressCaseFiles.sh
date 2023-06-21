#!/bin/bash

for i in Present*Case*/; do
    echo --\ working on\ ${i%/}
    zip -r "${i%/}.zip" "$i";
    rm -rf ${i%/}
done
