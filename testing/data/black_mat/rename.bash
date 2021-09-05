for IDX in 2 3 4 5 6 7 8 9 10
do
    file_src=BG_black_piece_${IDX}.png
    file_target=BG_black_piece_$((${IDX}-1)).png

    echo ${file_src}
    echo ${file_target}
    mv ${file_src} ${file_target}

    echo ""
done
