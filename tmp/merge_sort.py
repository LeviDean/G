def merge_sort(arr):
    """
    Merge Sort Algorithm
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        arr: List of comparable elements to sort
    
    Returns:
        Sorted list in ascending order
    """
    # Base case: arrays with 0 or 1 element are already sorted
    if len(arr) <= 1:
        return arr
    
    # Divide the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Recursively sort both halves
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)
    
    # Merge the sorted halves
    return merge(left_sorted, right_sorted)


def merge(left, right):
    """
    Merge two sorted arrays into one sorted array
    
    Args:
        left: Sorted list
        right: Sorted list
    
    Returns:
        Merged sorted list
    """
    result = []
    left_idx = 0
    right_idx = 0
    
    # Compare elements from both arrays and add smaller one to result
    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] <= right[right_idx]:
            result.append(left[left_idx])
            left_idx += 1
        else:
            result.append(right[right_idx])
            right_idx += 1
    
    # Add remaining elements from left array (if any)
    while left_idx < len(left):
        result.append(left[left_idx])
        left_idx += 1
    
    # Add remaining elements from right array (if any)
    while right_idx < len(right):
        result.append(right[right_idx])
        right_idx += 1
    
    return result


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 4, 6, 1, 3],
        [1],
        [],
        [3, 3, 3, 3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    print("Merge Sort Algorithm Demo")
    print("=" * 30)
    
    for i, arr in enumerate(test_arrays, 1):
        original = arr.copy()
        sorted_arr = merge_sort(arr)
        print(f"Test {i}:")
        print(f"Original: {original}")
        print(f"Sorted:   {sorted_arr}")
        print()